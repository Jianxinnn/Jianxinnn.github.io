import Link from "next/link";
import { BlogSourceMark } from "@/components/blog-source-mark";
import { BlogTags } from "@/components/blog-tags";
import { getBlogArchive } from "@/content/blog/posts";
import type { BlogPost } from "@/content/blog/posts";
import { formatDate } from "@/lib/content";

export const metadata = {
  title: "Blog Archive"
};

function PostTitleLink({ post }: { post: BlogPost }) {
  if (post.href.startsWith("http")) {
    return (
      <a href={post.href} rel="noreferrer" target="_blank">
        {post.title}
      </a>
    );
  }

  return <Link href={post.href}>{post.title}</Link>;
}

export default function BlogArchivePage() {
  const archive = getBlogArchive();

  return (
    <div className="blog-page">
      <header className="blog-index-header">
        <p className="eyebrow">Blog archive</p>
        <h1>All notes by year.</h1>
        <p>Static archive for scanning older notes as the site grows.</p>
        <Link className="back-link" href="/blog">
          Blog
        </Link>
      </header>

      <div className="blog-archive-list">
        {archive.map((group) => (
          <section className="blog-archive-year" id={`year-${group.year}`} key={group.year}>
            <h2>{group.year}</h2>
            {group.posts.map((post) => (
              <article className="blog-archive-post" key={post.slug}>
                <h3>
                  <PostTitleLink post={post} />
                </h3>
                <BlogSourceMark source={post.source} />
                <p>{post.summary}</p>
                <div className="entry-meta">
                  <time dateTime={post.date}>{formatDate(post.date)}</time>
                  <span aria-hidden="true">·</span>
                  <span>{post.readingTime}</span>
                  {post.category ? (
                    <>
                      <span aria-hidden="true">·</span>
                      <span>{post.category}</span>
                    </>
                  ) : null}
                </div>
                <BlogTags tags={post.tags} />
              </article>
            ))}
          </section>
        ))}
      </div>
    </div>
  );
}
