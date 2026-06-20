import Link from "next/link";
import { slugifyTag } from "@/content/blog/posts";

type BlogTagsProps = {
  tags?: string[];
};

export function BlogTags({ tags }: BlogTagsProps) {
  if (!tags?.length) {
    return null;
  }

  return (
    <div className="blog-tags" aria-label="Tags">
      {tags.map((tag) => (
        <Link href={`/blog/tags/${slugifyTag(tag)}`} key={tag}>
          {tag}
        </Link>
      ))}
    </div>
  );
}
