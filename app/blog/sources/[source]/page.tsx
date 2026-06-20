import Link from "next/link";
import { notFound } from "next/navigation";
import { BlogList } from "@/components/blog-list";
import {
  getBlogSourceStats,
  getBlogSourceStatus,
  getPostsBySourceStatus
} from "@/content/blog/posts";
import type { BlogSourceStatus } from "@/content/blog/posts";

type BlogSourcePageProps = {
  params: Promise<{
    source: string;
  }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  return getBlogSourceStats().map((source) => ({
    source: source.status
  }));
}

export async function generateMetadata({ params }: BlogSourcePageProps) {
  const { source } = await params;
  const currentSource = getBlogSourceStatus(source);

  return {
    title: currentSource ? `Blog: ${currentSource.title}` : "Blog"
  };
}

export default async function BlogSourcePage({ params }: BlogSourcePageProps) {
  const { source } = await params;
  const currentSource = getBlogSourceStatus(source);

  if (!currentSource) {
    notFound();
  }

  const posts = getPostsBySourceStatus(currentSource.status as BlogSourceStatus);

  if (posts.length === 0) {
    notFound();
  }

  return (
    <div className="blog-page">
      <header className="blog-index-header">
        <p className="eyebrow">Source</p>
        <h1>{currentSource.title}</h1>
        <p>
          {currentSource.description} {posts.length}{" "}
          {posts.length === 1 ? "post" : "posts"} in this source group.
        </p>
        <Link className="back-link" href="/blog">
          Blog
        </Link>
      </header>
      <BlogList posts={posts} />
    </div>
  );
}
