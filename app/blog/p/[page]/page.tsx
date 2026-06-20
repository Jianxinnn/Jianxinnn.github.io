import Link from "next/link";
import { notFound } from "next/navigation";
import { BlogList } from "@/components/blog-list";
import { BlogPagination } from "@/components/blog-pagination";
import { BlogSearch } from "@/components/blog-search";
import { getBlogPage } from "@/content/blog/posts";

type BlogPageProps = {
  params: Promise<{
    page: string;
  }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  const { totalPages } = getBlogPage(1);

  return Array.from({ length: Math.max(0, totalPages - 1) }, (_, index) => ({
    page: String(index + 2)
  }));
}

export async function generateMetadata({ params }: BlogPageProps) {
  const { page } = await params;

  return {
    title: `Blog Page ${page}`
  };
}

export default async function PaginatedBlogPage({ params }: BlogPageProps) {
  const pageNumber = Number((await params).page);

  if (!Number.isInteger(pageNumber) || pageNumber < 2) {
    notFound();
  }

  const page = getBlogPage(pageNumber);

  if (page.currentPage !== pageNumber) {
    notFound();
  }

  return (
    <div className="blog-page">
      <header className="blog-index-header">
        <p className="eyebrow">Blog</p>
        <h1>Research notes, illustrated explainers, and build logs.</h1>
        <p>
          Page {page.currentPage} of {page.totalPages}. More research notes,
          illustrated explainers, and build logs.
        </p>
        <Link className="back-link" href="/blog">
          First page
        </Link>
      </header>
      <BlogSearch />
      <BlogList posts={page.posts} />
      <BlogPagination currentPage={page.currentPage} totalPages={page.totalPages} />
    </div>
  );
}
