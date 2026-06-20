import Link from "next/link";
import { notFound } from "next/navigation";
import { BlogList } from "@/components/blog-list";
import { getBlogCategories, getPostsByCategorySlug } from "@/content/blog/posts";

type BlogCategoryPageProps = {
  params: Promise<{
    category: string;
  }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  return getBlogCategories().map((category) => ({
    category: category.slug
  }));
}

export async function generateMetadata({ params }: BlogCategoryPageProps) {
  const { category } = await params;
  const currentCategory = getBlogCategories().find((item) => item.slug === category);

  return {
    title: currentCategory ? `Blog: ${currentCategory.label}` : "Blog"
  };
}

export default async function BlogCategoryPage({ params }: BlogCategoryPageProps) {
  const { category } = await params;
  const currentCategory = getBlogCategories().find((item) => item.slug === category);
  const posts = getPostsByCategorySlug(category);

  if (!currentCategory || posts.length === 0) {
    notFound();
  }

  return (
    <div className="blog-page">
      <header className="blog-index-header">
        <p className="eyebrow">Category</p>
        <h1>{currentCategory.label}</h1>
        <p>{currentCategory.description}</p>
        <Link className="back-link" href="/blog">
          Blog
        </Link>
      </header>
      <BlogList posts={posts} />
    </div>
  );
}
