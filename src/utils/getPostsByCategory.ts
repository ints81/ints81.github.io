import type { CollectionEntry } from "astro:content";
import getSortedPosts from "./getSortedPosts";
import { slugifyStr } from "./slugify";

/**
 * 부모 카테고리에 속한 포스트 조회 (직속 + 모든 하위 카테고리)
 */
const getPostsByParentCategory = (
  posts: CollectionEntry<"blog">[],
  parentSlug: string
) => {
  const filtered = posts.filter(post => {
    const slug = slugifyStr(post.data.category.parent);
    return slug === parentSlug;
  });
  return getSortedPosts(filtered);
};

/**
 * 부모+자식 카테고리에 속한 포스트 조회
 */
const getPostsByCategory = (
  posts: CollectionEntry<"blog">[],
  parentSlug: string,
  childSlug: string
) => {
  const filtered = posts.filter(post => {
    const pSlug = slugifyStr(post.data.category.parent);
    const cSlug = post.data.category.child
      ? slugifyStr(post.data.category.child)
      : null;
    return pSlug === parentSlug && cSlug === childSlug;
  });
  return getSortedPosts(filtered);
};

/**
 * 부모 카테고리만 있는 포스트 (자식 없는 직속 포스트)
 */
const getDirectPostsByParentCategory = (
  posts: CollectionEntry<"blog">[],
  parentSlug: string
) => {
  const filtered = posts.filter(post => {
    const slug = slugifyStr(post.data.category.parent);
    const hasChild = !!post.data.category.child;
    return slug === parentSlug && !hasChild;
  });
  return getSortedPosts(filtered);
};

export { getPostsByParentCategory, getPostsByCategory, getDirectPostsByParentCategory };
export default getPostsByCategory;
