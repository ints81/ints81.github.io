---
title: vLLM Attention 코드 분석 (작성 중)
date: 2024-06-13 23:13:43 +0900
categories: [Deep Learning, Frameworks, vLLM]
tags: [vLLM, Paged Attention]     # TAG names should always be lowercase
toc: true
comments: true
---

vLLM의 paged attention kernel 코드를 분석해서 적어놓는다. v1 먼저 적고 그 다음에 v2를 적을 생각이다. 다루는 코드는 vLLM v0.5.0이다.



## Paged Attention V1

vLLM의 paged attention은 pybind를 통해 paged attention의 C++/CUDA 구현을 Python에서 사용할 수 있다. 해당 내용은 글의 주제를 넘어가기 때문에 소개하지 않고, 여기서는 C++ 구현 중 어디에서 paged attention v1을 호출하는지부터 간단히 짚고 넘어간다.

paged attention v1은 다음 부분에서 호출된다. 불필요한 코드는 줄이고 나타냈다.

```c++
// vllm/csrc/attention/attention_kernels.cu, line 690
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, float kv_scale,
    const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  ...

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len =
      DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    ...
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}
```

`paged_attention_v1_launcher`는 `LAUNCH_PAGED_ATTENTION_V1`이라는 매크로를 호출하는데 해당 매크로의 구현은 아래와 같다.

```c++
// vllm/csrc/attention/attention_kernels.cu, line 673
#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, IS_BLOCK_SPARSE>),  \
      shared_mem_size);                                                     \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>   \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          kv_scale, tp_rank, blocksparse_local_blocks,                      \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);
```

위의 매크로는 결과적으로 `paged_attention_v1_kernel`이라는 CUDA kernel을 호출하는데, 매크로 특성 상 매크로 내용물을 호출하는 위치에 치환한 뒤에 컴파일을 하기 때문에 이 kernel을 호출할 때 사용하는 grid size와 block size는 `paged_attention_v1_launcher`의 `grid`와 `block` 변수이다. 따라서 paged attention v1 kernel의

> grid size : (number of heads, number of tokens, 1)
>
> block size : (number of threads, 1, 1), number of threads는 기본값이 128이다. {: .prompt-info}

라는 걸 확인 할 수 있다.

`paged_attention_v1_kernel`의 구현부를 간단하게 보면

```c++
// vllm/csrc/attention/attention_kernels.cu, line 499
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_kernel(
    ...
    const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE>(
      ...
      blocksparse_head_sliding_step);
}

```

이 함수가 최종적으로는 **`paged_attention_kernel`**이라는 CUDA kernel을 호출하는 것을 알 수 있다. **이 kernel이 실질적인 paged attention에 대한 구현**이다. 

`paged_attention_kernel`의 구현은 아래와 같다. 이 kernel은 paged attention v1과 paged attention v2가 합쳐져있는 구현으로, 각 인자에 어떤 값이 들어가는지에 따라 v1으로 동작하거나 v2로 동작한다.



vLLM의 paged attention의 C++ 구현의 함수 시그니처는 다음과 같다. C++ 구현이 v1과 v2가 하나로 합쳐져서 구현이 되어있다. 따라서 인자에 뭐가 들어가는지에 따라 v1 kernel로 동작할 수도 있고, v2 kernel로 동작할 수도 있다. 

```c++
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step)
```

각 인자의 의미는 다음과 같다.

1. exp_sums : paged attention v2 전용이고, v1에서는 nullptr.
2. max_logits : paged attention v2 전용이고, v1에서는 nullptr.
3. out : paged attention의 결과를 받는 인자다.
4. q, k_cache, v_cache : q는 input을 한 번 matmul을 거친 query, k_cache와 v_cache는 input에서 matmul을 거친 후에 caching을 한 k와 v이다.
5. num_kv_heads : 
6. scale : 
7. block_tables : 
8. seq_lens : 
9. max_num_blocks_per_seq : 
10. alibi_slopes : 
11. q_stride, kv_block_stride, kv_head_stride : 
12. kv_scale : 
13. tp_rank : 
14. blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size, block_sparse_head_sliding_step : 
