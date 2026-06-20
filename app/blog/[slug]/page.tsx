import Link from "next/link";
import { notFound } from "next/navigation";
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

  return (
    <article className="blog-article-page">
      <header className="blog-article-header">
        <Link className="back-link" href="/blog">
          Blog
        </Link>
        <h1>{post.title}</h1>
        <p>{post.summary}</p>
        <div className="entry-meta">
          <span>{post.readingTime}</span>
          <span aria-hidden="true">·</span>
          <time dateTime={post.date}>{formatDate(post.date)}</time>
        </div>
        {post.tags?.length ? (
          <div className="blog-tags" aria-label="Tags">
            {post.tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
        ) : null}
      </header>
      <div className="mdx-body">
        <Content />
      </div>
    </article>
  );
}
