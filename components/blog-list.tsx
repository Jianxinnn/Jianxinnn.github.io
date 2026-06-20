import Image from "next/image";
import Link from "next/link";
import type { ReactNode } from "react";
import type { BlogPost } from "@/content/blog/posts";
import { formatDate } from "@/lib/content";

type BlogListProps = {
  posts: BlogPost[];
  showImages?: boolean;
};

function PostLink({
  children,
  className,
  post
}: {
  children: ReactNode;
  className?: string;
  post: BlogPost;
}) {
  if (post.href.startsWith("http")) {
    return (
      <a className={className} href={post.href} rel="noreferrer" target="_blank">
        {children}
      </a>
    );
  }

  return (
    <Link className={className} href={post.href}>
      {children}
    </Link>
  );
}

export function BlogList({ posts, showImages = true }: BlogListProps) {
  return (
    <div className="blog-feed">
      {posts.map((post) => (
        <article className="blog-card" key={post.slug}>
          <div className="blog-card-copy">
            <div className="blog-title-line">
              <PostLink className="blog-card-title" post={post}>
                {post.title}
              </PostLink>
              {post.badge ? <span className="blog-badge">{post.badge}</span> : null}
            </div>
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
          </div>
          <PostLink className="blog-card-year" post={post}>
            {new Date(post.date).getFullYear()}
          </PostLink>
          {showImages && post.image ? (
            <PostLink className="blog-card-image-link" post={post}>
              <Image
                alt=""
                className="blog-card-image"
                height={214}
                src={post.image}
                width={320}
              />
            </PostLink>
          ) : null}
        </article>
      ))}
    </div>
  );
}
