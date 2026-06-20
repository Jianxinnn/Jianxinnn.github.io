import Link from "next/link";
import { notFound } from "next/navigation";
import { BlogList } from "@/components/blog-list";
import { getBlogTags, getPostsByTagSlug } from "@/content/blog/posts";

type BlogTagPageProps = {
  params: Promise<{
    tag: string;
  }>;
};

export function generateStaticParams() {
  return getBlogTags().map((tag) => ({
    tag: tag.slug
  }));
}

export async function generateMetadata({ params }: BlogTagPageProps) {
  const { tag } = await params;
  const currentTag = getBlogTags().find((item) => item.slug === tag);

  return {
    title: currentTag ? `Blog: ${currentTag.label}` : "Blog"
  };
}

export default async function BlogTagPage({ params }: BlogTagPageProps) {
  const { tag } = await params;
  const currentTag = getBlogTags().find((item) => item.slug === tag);
  const posts = getPostsByTagSlug(tag);

  if (!currentTag || posts.length === 0) {
    notFound();
  }

  return (
    <div className="blog-page">
      <header className="blog-index-header">
        <p className="eyebrow">Tag</p>
        <h1>{currentTag.label}</h1>
        <p>
          {currentTag.count} {currentTag.count === 1 ? "post" : "posts"} in this
          topic.
        </p>
        <Link className="back-link" href="/blog">
          Blog
        </Link>
      </header>
      <BlogList posts={posts} />
    </div>
  );
}
