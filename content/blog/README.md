# Blog Content Guide

The blog system has one public index and multiple source formats. Every post
must be registered in `content/blog/posts.ts`; the source format only changes
how the detail page is rendered.

## Standard MDX Post

Use this for new writing.

1. Create `content/blog/my-post.mdx`.
2. Import it in `content/blog/content.ts` and add it to `blogPostContent`.
3. Add a `BlogPost` record in `content/blog/posts.ts`:

```ts
{
  slug: "my-post",
  title: "My Post",
  summary: "One sentence summary.",
  date: "2026-06-20",
  readingTime: "5 min read",
  sourceType: "mdx",
  href: "/blog/my-post/",
  image: "/assets/visuals/notes-field.png",
  badge: "转载 / 译",
  tags: ["tag"]
}
```

The post will appear on `/blog`, Home, and Archive automatically.
Use `badge` only when the card needs a small source-status label, such as
`"转载"` or `"转载 / 译"`.

## Imported HTML Post

Use this when the article is already a full standalone HTML page.

1. Put the file at `public/blog/my-html-post/index.html`.
2. Add a `BlogPost` record with `sourceType: "html"` and
   `href: "/blog/my-html-post/"`.

The HTML page keeps its own article layout, while `/blog` still shows the same
title, summary, date, read time, tags, and thumbnail as other posts.

## External Post

Use this for posts hosted elsewhere.

```ts
{
  slug: "external-post",
  title: "External Post",
  summary: "One sentence summary.",
  date: "2026-06-20",
  readingTime: "8 min read",
  sourceType: "external",
  href: "https://example.com/post",
  tags: ["tag"]
}
```

External posts appear in the index and open in a new tab.
