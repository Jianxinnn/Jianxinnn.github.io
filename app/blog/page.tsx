import Link from "next/link";
import { BlogList } from "@/components/blog-list";
import { BlogPagination } from "@/components/blog-pagination";
import { BlogSearch } from "@/components/blog-search";
import {
  blogPosts,
  getBlogArchive,
  getBlogCategories,
  getBlogPage,
  getBlogSourceStats,
  getBlogTags,
  getFeaturedPosts
} from "@/content/blog/posts";
import { formatDate } from "@/lib/content";

export const metadata = {
  title: "Blog"
};

export default function BlogPage() {
  const page = getBlogPage(1);
  const archive = getBlogArchive();
  const categories = getBlogCategories();
  const featuredPosts = getFeaturedPosts().slice(0, 1);
  const featuredSlugs = new Set(featuredPosts.map((post) => post.slug));
  const latestPost = page.posts[0];
  const listPosts = page.posts.filter((post) => !featuredSlugs.has(post.slug));
  const sourceStats = getBlogSourceStats();
  const tags = getBlogTags();

  return (
    <div className="blog-page blog-library-page">
      <header className="blog-index-header">
        <p className="eyebrow">Blog</p>
        <h1>Research notes, illustrated explainers, and build logs.</h1>
        <p>
          A compact log of AI-for-science systems, paper readings, visual
          explanations, and research workflow experiments.
        </p>
      </header>
      <BlogSearch />
      <section className="blog-library-summary" aria-label="Blog summary">
        <div>
          <span>{blogPosts.length}</span>
          <p>Posts</p>
        </div>
        <div>
          <span>{categories.length}</span>
          <p>Categories</p>
        </div>
        <div>
          <span>{latestPost ? formatDate(latestPost.date) : "Draft"}</span>
          <p>Latest update</p>
        </div>
      </section>

      <div className="blog-library-layout">
        <div className="blog-library-main">
          {featuredPosts.length ? (
            <section className="blog-featured-section" aria-labelledby="featured-posts">
              <div className="section-heading">
                <h2 id="featured-posts">Featured</h2>
              </div>
              <BlogList posts={featuredPosts} />
            </section>
          ) : null}

          <section aria-labelledby="all-posts">
            <div className="section-heading">
              <h2 id="all-posts">All posts</h2>
              <Link href="/blog/archive">Archive</Link>
            </div>
            <BlogList posts={listPosts.length ? listPosts : page.posts} />
            <BlogPagination currentPage={page.currentPage} totalPages={page.totalPages} />
          </section>
        </div>

        <aside className="blog-library-rail" aria-label="Blog filters">
          {categories.length ? (
            <section className="blog-rail-block">
              <h2>Categories</h2>
              <nav>
                {categories.map((category) => (
                  <Link href={`/blog/categories/${category.slug}`} key={category.slug}>
                    <span>{category.label}</span>
                    <span>{category.count}</span>
                  </Link>
                ))}
              </nav>
            </section>
          ) : null}

          {tags.length ? (
            <section className="blog-rail-block">
              <h2>Tags</h2>
              <nav className="blog-tag-cloud">
                {tags.map((tag) => (
                  <Link href={`/blog/tags/${tag.slug}`} key={tag.slug}>
                    {tag.label}
                    <span>{tag.count}</span>
                  </Link>
                ))}
              </nav>
            </section>
          ) : null}

          {sourceStats.length ? (
            <section className="blog-rail-block">
              <h2>Sources</h2>
              <div className="blog-source-stats">
                {sourceStats.map((item) => (
                  <div key={item.status}>
                    <span>{item.label}</span>
                    <span>{item.count}</span>
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          {archive.length ? (
            <section className="blog-rail-block">
              <h2>Years</h2>
              <nav>
                {archive.map((group) => (
                  <Link href={`/blog/archive#year-${group.year}`} key={group.year}>
                    <span>{group.year}</span>
                    <span>{group.posts.length}</span>
                  </Link>
                ))}
              </nav>
            </section>
          ) : null}
        </aside>
      </div>
    </div>
  );
}
