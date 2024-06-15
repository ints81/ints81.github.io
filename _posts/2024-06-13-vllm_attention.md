---
title: vLLM Attention 코드 분석
date: 2024-06-13 23:13:43 +0900
categories: [Deep Learning, Frameworks, vLLM]
tags: [vLLM, Paged Attention]     # TAG names should always be lowercase
toc: true
comments: true
---

vLLM의 paged attention의 C++ 구현의 함수 시그니처는 다음과 같다. C++ 구현이 v1과 v2가 하나로 합쳐져서 구현이 되어있다. 따라서 인자에 뭐가 들어가는지에 따라 v1 kernel로 동작할 수도 있고, v2 kernel로 동작할 수도 있다. 

```cpp
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
