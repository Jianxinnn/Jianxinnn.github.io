import Link from "next/link";
import { notFound } from "next/navigation";
import { BlogSectionNav } from "@/components/blog-section-nav";
import { BlogSourceMark } from "@/components/blog-source-mark";
import { BlogTags } from "@/components/blog-tags";
import { ViewCountBadge } from "@/components/view-count-badge";
import { blogPostContent } from "@/content/blog/content";
import { getBlogPost, sortBlogPosts } from "@/content/blog/posts";
import { formatDate } from "@/lib/content";

type BlogPostPageProps = {
  params: Promise<{
    slug: string;
  }>;
};

export function generateStaticParams() {
  return sortBlogPosts()
    .filter((post) => post.sourceType === "mdx")
    .map((post) => ({
      slug: post.slug
    }));
}

export async function generateMetadata({ params }: BlogPostPageProps) {
  const { slug } = await params;
  const post = getBlogPost(slug);

  if (!post) {
    return {
      title: "Blog"
    };
  }

  return {
    title: post.title,
    description: post.summary
  };
}

export default async function BlogPostPage({ params }: BlogPostPageProps) {
  const { slug } = await params;
  const post = getBlogPost(slug);

  const Content = post ? blogPostContent[post.slug] : undefined;

  if (!post || post.sourceType !== "mdx" || !Content) {
    notFound();
  }

  const tags = post.tags ?? [];
  const articleClassName = [
    "blog-article-page",
    `blog-article-${post.slug}`,
    post.language === "bilingual" ? "is-bilingual-article" : undefined,
    tags.includes("illustrated note") ? "is-illustrated-note" : undefined
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <article className={articleClassName}>
      <header className="blog-article-header">
        <Link className="back-link" href="/blog">
          Blog
        </Link>
        <div className="blog-article-title-line">
          <h1>{post.title}</h1>
          {post.badge ? <span className="blog-badge">{post.badge}</span> : null}
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
        <BlogSourceMark source={post.source} variant="article" />
        <BlogTags tags={post.tags} />
      </header>
      <div className="mdx-body">
        <Content />
      </div>
      <BlogSectionNav />
    </article>
  );
}
