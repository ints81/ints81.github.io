import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

/**
 * 시리즈 slug에 해당하는 포스트 조회 (order 기준 정렬)
 */
const getPostsBySeries = (
  posts: CollectionEntry<"blog">[],
  seriesSlug: string
): CollectionEntry<"blog">[] => {
  const filtered = posts
    .filter(postFilter)
    .filter(post => {
      const series = post.data.series;
      if (!series) return false;
      return slugifyStr(series.name) === seriesSlug;
    })
    .sort((a, b) => {
      const orderA = a.data.series!.order;
      const orderB = b.data.series!.order;
      return orderA - orderB;
    });

  return filtered;
};

export default getPostsBySeries;
