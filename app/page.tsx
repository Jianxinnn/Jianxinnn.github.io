import Link from "next/link";
import { Mail } from "lucide-react";
import { BlogList } from "@/components/blog-list";
import { BlogPostImage } from "@/components/blog-post-image";
import { BlogSourceMark } from "@/components/blog-source-mark";
import { blogPosts, sortBlogPosts } from "@/content/blog/posts";
import { profile } from "@/content/profile";
import { formatDate } from "@/lib/content";

export default function HomePage() {
  const posts = sortBlogPosts(blogPosts);
  const currentWork = posts.slice(0, 2);
  const history = posts.slice(2);

  return (
    <div className="page-shell">
      <section className="home-hero">
        <div className="home-signature" aria-label={profile.name}>
          <span className="signature-letter" aria-hidden="true">J</span>
          <h1>{profile.name}</h1>
        </div>

        <section className="current-panel" aria-labelledby="current-work-heading">
          <div className="current-heading">
            <div>
              <p className="eyebrow">Current work</p>
              <h1 id="current-work-heading">Research systems and working notes</h1>
            </div>
            <Link href="/blog">Blog</Link>
          </div>

          <div className="current-work-list">
            {currentWork.map((post, index) => (
              <article className="current-work-item" key={post.slug}>
                {post.image ? (
                  post.href.startsWith("http") ? (
                    <a
                      aria-label={post.title}
                      className="current-work-image-link"
                      href={post.href}
                      rel="noreferrer"
                      target="_blank"
                    >
                      <BlogPostImage
                        alt=""
                        className="current-work-image"
                        height={214}
                        priority={index === 0}
                        src={post.image}
                        width={320}
                      />
                    </a>
                  ) : (
                    <Link aria-label={post.title} className="current-work-image-link" href={post.href}>
                      <BlogPostImage
                        alt=""
                        className="current-work-image"
                        height={214}
                        priority={index === 0}
                        src={post.image}
                        width={320}
                      />
                    </Link>
                  )
                ) : null}
                <div className="current-work-copy">
                  <div className="entry-meta">
                    <time dateTime={post.date}>{formatDate(post.date)}</time>
                    <span aria-hidden="true">·</span>
                    <span>{post.readingTime}</span>
                  </div>
                  <BlogSourceMark source={post.source} />
                  <div className="current-title-line">
                    <h2>
                      {post.href.startsWith("http") ? (
                        <a href={post.href} rel="noreferrer" target="_blank">
                          {post.title}
                        </a>
                      ) : (
                        <Link href={post.href}>{post.title}</Link>
                      )}
                    </h2>
                    {post.badge ? <span className="blog-badge">{post.badge}</span> : null}
                  </div>
                  <p>{post.summary}</p>
                </div>
              </article>
            ))}
          </div>
        </section>
      </section>

      <section className="home-log-layout">
        <div className="recent-log">
          <div className="section-heading">
            <h2>Recent log</h2>
            <Link href="/blog">Blog</Link>
          </div>
          <BlogList posts={history} />
        </div>

        <aside className="mail-panel" aria-label="Email updates">
          <div className="mail-panel-icon" aria-hidden="true">
            <Mail size={20} strokeWidth={2} />
          </div>
          <h2>Mail</h2>
          <p>Occasional updates on research systems, paper notes, and project logs.</p>
          <form action={`mailto:${profile.email}`} className="mail-form">
            <input aria-label="Email address" name="email" placeholder="Email address" type="email" />
            <button type="submit">Subscribe</button>
          </form>
        </aside>
      </section>
    </div>
  );
}
