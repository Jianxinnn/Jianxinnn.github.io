# Personal Profile

A publication-style personal profile site inspired by a Substack research publication layout.

## Structure

- `app/` contains routes and page shells.
- `components/` contains shared layout primitives.
- `content/blog/posts/` contains one directory per blog post.
- `.generated/` contains the generated blog index and MDX content map.
- `scripts/` contains content generation and validation scripts.
- `content/profile.ts` contains personal identity, links, and About copy.
- `content/entries.ts` aggregates blog posts, projects, notes, talks, and publications.
- `lib/content.ts` contains small helpers for sorting and grouping entries.

## Local Development

```bash
npm install
npm run content:generate
npm run dev
```

## GitHub Pages

The project is configured for static export with `next build`. GitHub Actions builds the site, generates the Pagefind search index, and deploys the generated `out/` directory to GitHub Pages.

## Update Content

Edit `content/profile.ts` for identity and About sections. Add formal writing under `content/blog/posts/<slug>/` with a `meta.ts` file; generated metadata automatically appears on Home, Blog, Archive, tag pages, and search. Use `content/entries.ts` only for non-blog notes, projects, talks, or other archive items.

See `content/blog/README.md` for the MDX, imported HTML, external post, source-filter, and bilingual long-post workflow.

Quick start for a new blog post:

```bash
npm run blog:new -- --title="My New Post" --slug="my-new-post"
npm run content:generate
npm run content:validate
```

For reposts or translations, scaffold the source metadata directly:

```bash
npm run blog:new -- --title="My Translation" --slug="my-translation" --language="zh" --category="Paper notes" --source-status="translation"
```
