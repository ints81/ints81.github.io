import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

export interface SeriesInfo {
  name: string;
  slug: string;
  posts: { post: CollectionEntry<"blog">; order: number }[];
}

/**
 * 포스트 목록에서 시리즈 정보 추출.
 * series 필드가 있는 포스트만 그룹화하여 반환.
 */
const getSeriesFromPosts = (
  posts: CollectionEntry<"blog">[]
): SeriesInfo[] => {
  const filtered = posts.filter(postFilter);
  const seriesMap = new Map<string, { name: string; posts: { post: CollectionEntry<"blog">; order: number }[] }>();

  for (const post of filtered) {
    const series = post.data.series;
    if (!series) continue;

    const slug = slugifyStr(series.name);
    if (!seriesMap.has(slug)) {
      seriesMap.set(slug, { name: series.name, posts: [] });
    }
    const entry = seriesMap.get(slug)!;
    entry.posts.push({ post, order: series.order });
  }

  return Array.from(seriesMap.entries())
    .map(([slug, { name, posts }]) => ({
      name,
      slug,
      posts: posts.sort((a, b) => a.order - b.order),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
};

export default getSeriesFromPosts;
