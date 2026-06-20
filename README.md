# Personal Profile

A publication-style personal profile site inspired by a Substack research publication layout.

## Structure

- `app/` contains routes and page shells.
- `components/` contains shared layout primitives.
- `content/profile.ts` contains personal identity, links, and About copy.
- `content/entries.ts` contains posts, projects, notes, talks, and publications.
- `lib/content.ts` contains small helpers for sorting and grouping entries.

## Local Development

```bash
npm install
npm run dev
```

## Update Content

Edit `content/profile.ts` for identity and About sections. Edit `content/entries.ts` to add profile entries. New entries automatically appear on Home, Notes, and Archive based on their `type` and `featured` values.
