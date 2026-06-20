# Blog Content Guide

Every blog post lives in its own directory under `content/blog/posts/`.
Metadata is written once in `meta.ts`; generated files feed Home, Blog, Archive,
tag pages, and MDX detail routes.

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
  tags: ["research systems"]
} satisfies BlogPostMeta;

export default meta;
```

The generator reads `index.mdx`, estimates `readingTime`, and creates the
route `/blog/my-post/`.

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
  tags: ["research systems"]
} satisfies BlogPostMeta;
```

## Commands

```bash
npm run content:generate
npm run content:validate
npm run typecheck
npm run build
```

`npm run build` generates metadata, validates content, builds the static site,
and then creates the Pagefind search index under `out/pagefind/`.

## Tags And Badges

Allowed tags live in `content/blog/tags.ts`. Add a tag there before using it in
`meta.ts`.

Use `badge` only when the post needs a source-status note:

```ts
badge: "转载 / 译"
```
