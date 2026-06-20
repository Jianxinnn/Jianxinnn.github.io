import { BlogList } from "@/components/blog-list";
import { blogPosts, sortBlogPosts } from "@/content/blog/posts";

export const metadata = {
  title: "Blog"
};

export default function BlogPage() {
  const posts = sortBlogPosts(blogPosts);

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
      <BlogList posts={posts} />
    </div>
  );
}
