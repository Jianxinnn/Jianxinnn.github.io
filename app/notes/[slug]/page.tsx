import Link from "next/link";
import { notFound } from "next/navigation";
import { ViewCountBadge } from "@/components/view-count-badge";
import { blogPostContent } from "@/content/blog/content";
import { getBlogPost } from "@/content/blog/posts";
import { entries } from "@/content/entries";
import { formatDate } from "@/lib/content";

type NotePageProps = {
  params: Promise<{
    slug: string;
  }>;
};

const notes = entries.filter((entry) => entry.type === "note");

export const dynamicParams = false;

export function generateStaticParams() {
  return notes.map((note) => ({
    slug: note.slug
  }));
}

export async function generateMetadata({ params }: NotePageProps) {
  const { slug } = await params;
  const note = notes.find((entry) => entry.slug === slug);

  if (!note) {
    return {
      title: "Notes"
    };
  }

  return {
    title: note.title,
    description: note.summary
  };
}

export default async function NotePage({ params }: NotePageProps) {
  const { slug } = await params;
  const note = notes.find((entry) => entry.slug === slug);
  const post = getBlogPost(slug);
  const Content = post ? blogPostContent[post.slug] : undefined;

  if (!note || !post || post.sourceType !== "mdx" || !Content) {
    notFound();
  }

  return (
    <article className={`blog-article-page note-article-page note-article-${note.slug}`}>
      <header className="blog-article-header">
        <Link className="back-link" href="/notes">
          Notes
        </Link>
        <div className="blog-article-title-line">
          <h1>{note.title}</h1>
        </div>
        <p>{note.summary}</p>
        <div className="entry-meta">
          <time dateTime={note.date}>{formatDate(note.date)}</time>
          <span aria-hidden="true">·</span>
          <span>Note</span>
          <span aria-hidden="true">·</span>
          <ViewCountBadge scope="note" slug={note.slug} />
        </div>
      </header>
      <div className="mdx-body" data-pagefind-body>
        <Content />
      </div>
    </article>
  );
}
