import type { BlogPostSource } from "@/content/blog/posts";

type BlogSourceMarkProps = {
  className?: string;
  source?: BlogPostSource;
  variant?: "compact" | "article";
};

const statusLabels = {
  imported: "Imported",
  original: "Original",
  repost: "Repost",
  translation: "Translation"
} satisfies Record<string, string>;

export function BlogSourceMark({
  className = "",
  source,
  variant = "compact"
}: BlogSourceMarkProps) {
  if (!source || source.status === "original") {
    return null;
  }

  const label = source.label ?? statusLabels[source.status];
  const classes = ["blog-source-mark", `blog-source-mark-${variant}`, className]
    .filter(Boolean)
    .join(" ");

  if (!source.originalUrl) {
    return <span className={classes}>{label}</span>;
  }

  return (
    <span className={classes}>
      <span>{label}</span>
      {variant === "article" ? <span aria-hidden="true">·</span> : null}
      <a href={source.originalUrl} rel="noreferrer" target="_blank">
        {variant === "article"
          ? `Original: ${source.originalTitle ?? source.originalUrl}`
          : "Original"}
      </a>
    </span>
  );
}
