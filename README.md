# Personal Profile

A publication-style personal profile site inspired by a Substack research publication layout.

## Structure

- `app/` contains routes and page shells.
- `components/` contains shared layout primitives.
- `content/blog/` contains blog metadata, MDX posts, and the blog content guide.
- `content/profile.ts` contains personal identity, links, and About copy.
- `content/entries.ts` aggregates blog posts, projects, notes, talks, and publications.
- `lib/content.ts` contains small helpers for sorting and grouping entries.

## Local Development

```bash
npm install
npm run dev
```

## GitHub Pages

The project is configured for static export with `next build`. GitHub Actions builds the site and deploys the generated `out/` directory to GitHub Pages.

## Update Content

Edit `content/profile.ts` for identity and About sections. Add formal writing in `content/blog/` and register it in `content/blog/posts.ts`; new blog posts automatically appear on Home, Blog, and Archive. Use `content/entries.ts` only for non-blog notes, projects, talks, or other archive items.

See `content/blog/README.md` for the MDX, imported HTML, and external post workflow.
