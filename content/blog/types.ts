export type BlogSourceType = "mdx" | "html" | "external" | "encrypted";
export type BlogLanguage = "en" | "zh" | "bilingual";
export type BlogSourceStatus = "original" | "translation" | "repost" | "imported";

export type BlogPostSource = {
  status: BlogSourceStatus;
  label?: string;
  originalTitle?: string;
  originalUrl?: string;
  note?: string;
};

export type BlogPostMeta = {
  title: string;
  summary: string;
  date: string;
  sourceType: BlogSourceType;
  href?: string;
  readingTime?: string;
  image?: string;
  badge?: string;
  category?: string;
  language?: BlogLanguage;
  source?: BlogPostSource;
  tags?: string[];
  featured?: boolean;
  listed?: boolean;
  updated?: string;
};

export type BlogPost = Required<
  Pick<BlogPostMeta, "title" | "summary" | "date" | "sourceType" | "href" | "readingTime">
> &
  Pick<
    BlogPostMeta,
    | "image"
    | "badge"
    | "category"
    | "language"
    | "source"
    | "tags"
    | "featured"
    | "listed"
    | "updated"
  > & {
    slug: string;
  };
