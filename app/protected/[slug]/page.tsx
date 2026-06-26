import Link from "next/link";
import { notFound } from "next/navigation";
import { BlogTags } from "@/components/blog-tags";
import { ProtectedArticle } from "@/components/protected-article";
import { ViewCountBadge } from "@/components/view-count-badge";
import { getProtectedPost, protectedPosts } from "@/content/protected-posts";
import { formatDate } from "@/lib/content";

const placeholderSlug = "_placeholder";

type ProtectedPostPageProps = {
  params: Promise<{
    slug: string;
  }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  const params = protectedPosts.map((post) => ({
    slug: post.slug
  }));

  return params.length ? params : [{ slug: placeholderSlug }];
}

export async function generateMetadata({ params }: ProtectedPostPageProps) {
  const { slug } = await params;
  const post = getProtectedPost(slug);

  if (!post) {
    return {
      title: "Protected article",
      robots: {
        follow: false,
        index: false
      }
    };
  }

  return {
    title: post.title,
    description: post.summary,
    robots: {
      follow: false,
      index: false
    }
  };
}

export default async function ProtectedPostPage({ params }: ProtectedPostPageProps) {
  const { slug } = await params;
  const post = getProtectedPost(slug);

  if (slug === placeholderSlug && !post) {
    return (
      <article className="blog-article-page protected-article-page">
        <header className="blog-article-header">
          <Link className="back-link" href="/blog">
            Blog
          </Link>
          <div className="blog-article-title-line">
            <h1>Protected article</h1>
            <span className="blog-badge">Protected</span>
          </div>
          <p>No protected article is configured yet.</p>
        </header>
      </article>
    );
  }

  if (!post || post.listed === false) {
    notFound();
  }

  return (
    <article className={`blog-article-page protected-article-page protected-article-${post.slug}`}>
      <header className="blog-article-header">
        <Link className="back-link" href="/blog">
          Blog
        </Link>
        <div className="blog-article-title-line">
          <h1>{post.title}</h1>
          <span className="blog-badge">{post.badge ?? "Protected"}</span>
        </div>
        <p>{post.summary}</p>
        <div className="entry-meta">
          <span>{post.readingTime}</span>
          <span aria-hidden="true">·</span>
          <time dateTime={post.date}>{formatDate(post.date)}</time>
          {post.category ? (
            <>
              <span aria-hidden="true">·</span>
              <span>{post.category}</span>
            </>
          ) : null}
          <span aria-hidden="true">·</span>
          <ViewCountBadge scope="blog" slug={post.slug} />
        </div>
        <BlogTags tags={post.tags} />
      </header>
      <ProtectedArticle post={post} />
    </article>
  );
}
