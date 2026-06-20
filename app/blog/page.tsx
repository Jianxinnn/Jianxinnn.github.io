import Link from "next/link";
import { BlogList } from "@/components/blog-list";
import { BlogPagination } from "@/components/blog-pagination";
import { BlogSearch } from "@/components/blog-search";
import { getBlogPage, getBlogTags } from "@/content/blog/posts";

export const metadata = {
  title: "Blog"
};

export default function BlogPage() {
  const page = getBlogPage(1);
  const tags = getBlogTags();

  return (
    <div className="blog-page">
      <header className="blog-index-header">
        <p className="eyebrow">Blog</p>
        <h1>Research notes, illustrated explainers, and build logs.</h1>
        <p>
          A compact log of AI-for-science systems, paper readings, visual
          explanations, and research workflow experiments.
        </p>
      </header>
      <BlogSearch />
      {tags.length ? (
        <nav aria-label="Blog tags" className="blog-tag-cloud">
          {tags.map((tag) => (
            <Link href={`/blog/tags/${tag.slug}`} key={tag.slug}>
              {tag.label}
              <span>{tag.count}</span>
            </Link>
          ))}
        </nav>
      ) : null}
      <BlogList posts={page.posts} />
      <BlogPagination currentPage={page.currentPage} totalPages={page.totalPages} />
    </div>
  );
}
