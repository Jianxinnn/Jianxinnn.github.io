# Blog Content Guide

Every blog post lives in its own directory under `content/blog/posts/`.
Metadata is written once in `meta.ts`; generated files feed Home, Blog,
category pages, archive pages, tag pages, search, and MDX detail routes.

The site is intentionally database-free. The filesystem is the source of truth;
`scripts/generate-blog-index.ts` turns post metadata into a typed static index.

## Standard MDX Post

Use this for new writing.

```txt
content/blog/posts/my-post/
  index.mdx
  meta.ts
```

`meta.ts`:

```ts
import type { BlogPostMeta } from "../../types";

const meta = {
  title: "My Post",
  summary: "One sentence summary.",
  date: "2026-06-21",
  sourceType: "mdx",
  image: "/assets/visuals/notes-field.png",
  category: "Build logs",
  language: "en",
  source: {
    status: "original"
  },
  tags: ["research systems"]
} satisfies BlogPostMeta;

export default meta;
```

The generator reads `index.mdx`, estimates `readingTime`, and creates the
route `/blog/my-post/`.

You can scaffold this structure with:

```bash
npm run blog:new -- --title="My Post" --slug="my-post"
```

## Imported HTML Post

Use this for full standalone HTML articles.

```txt
content/blog/posts/my-html-post/
  meta.ts
public/blog/my-html-post/
  index.html
```

The metadata still appears in Home, Blog, Archive, tags, and search. The HTML
page keeps its own article layout.

## External Post

Use this for content hosted elsewhere.

```ts
const meta = {
  title: "External Post",
  summary: "One sentence summary.",
  date: "2026-06-21",
  readingTime: "8 min read",
  sourceType: "external",
  href: "https://example.com/post",
  category: "Paper notes",
  language: "en",
  source: {
    status: "repost",
    label: "转载",
    originalTitle: "External Post",
    originalUrl: "https://example.com/post"
  },
  tags: ["research systems"]
} satisfies BlogPostMeta;
```

## Repost And Translation Marks

Use `source` for source status. The UI renders non-original posts as small gray
text and shows the original URL on MDX article pages.

```ts
source: {
  status: "translation",
  label: "转载 / 翻译",
  originalTitle: "The Original Title",
  originalUrl: "https://example.com/original"
}
```

Allowed `status` values:

- `original`
- `translation`
- `repost`
- `imported`

## Categories, Tags, And Images

Allowed categories live in `content/blog/categories.ts`. Allowed tags live in
`content/blog/tags.ts`. Add a value there before using it in `meta.ts`.

`image` can be either a local asset such as `/assets/visuals/notes-field.png`
or an external HTTPS image URL. Local assets should live under `public/assets/`.

## Bilingual Long Posts

For long translated posts, generate trusted segment data and render it with
`components/bilingual-article.tsx`. The component defaults to Chinese and lets
readers switch to English or side-by-side mode.

## Commands

```bash
npm run blog:new -- --title="My Post" --slug="my-post"
npm run content:generate
npm run content:validate
npm run typecheck
npm run build
```

`npm run build` generates metadata, validates content, builds the static site,
and then creates the Pagefind search index under `out/pagefind/`.

## Scale Plan

This static setup is the right default until the site has frequent multi-author
editing or hundreds of posts. When the archive grows, keep adding posts as
files; the generated index, Pagefind search, category pages, and yearly archive
continue to work without a database.
