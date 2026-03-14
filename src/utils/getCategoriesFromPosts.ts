import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

export interface CategoryNode {
  parent: string;
  parentSlug: string;
  children: { name: string; slug: string; postCount: number }[];
  directPostCount: number;
  totalPostCount: number;
}

/**
 * 포스트 목록에서 고유한 카테고리 트리 추출 (2-depth).
 * 새 글이 올라올 때 기존에 없던 카테고리도 자동으로 포함됨.
 */
const getCategoriesFromPosts = (
  posts: CollectionEntry<"blog">[]
): CategoryNode[] => {
  const filtered = posts.filter(postFilter);

  const parentMap = new Map<
    string,
    {
      children: Map<string, number>;
      directCount: number;
    }
  >();

  for (const post of filtered) {
    const { parent, child } = post.data.category;

    if (!parentMap.has(parent)) {
      parentMap.set(parent, { children: new Map(), directCount: 0 });
    }
    const node = parentMap.get(parent)!;

    if (child) {
      const count = node.children.get(child) ?? 0;
      node.children.set(child, count + 1);
    } else {
      node.directCount += 1;
    }
  }

  return Array.from(parentMap.entries())
    .map(([parent, { children, directCount }]) => {
      const childrenArray = Array.from(children.entries())
        .map(([name, postCount]) => ({
          name,
          slug: slugifyStr(name),
          postCount,
        }))
        .sort((a, b) => a.name.localeCompare(b.name));

      const childTotal = childrenArray.reduce((sum, c) => sum + c.postCount, 0);
      const totalPostCount = directCount + childTotal;

      return {
        parent,
        parentSlug: slugifyStr(parent),
        children: childrenArray,
        directPostCount: directCount,
        totalPostCount,
      };
    })
    .sort((a, b) => a.parent.localeCompare(b.parent));
};

export default getCategoriesFromPosts;
