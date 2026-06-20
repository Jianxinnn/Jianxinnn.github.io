import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const postsRoot = path.join(repoRoot, "content/blog/posts");

function getArg(name: string) {
  const prefix = `--${name}=`;
  return process.argv.find((arg) => arg.startsWith(prefix))?.slice(prefix.length);
}

function today() {
  return new Date().toISOString().slice(0, 10);
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

async function main() {
  const title = getArg("title");
  const slug = getArg("slug") ?? (title ? slugify(title) : undefined);

  if (!title || !slug) {
    throw new Error('Usage: npm run blog:new -- --title="Post title" --slug="post-slug"');
  }

  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
    throw new Error("Slug must use lowercase kebab-case.");
  }

  const postDir = path.join(postsRoot, slug);

  await fs.mkdir(postDir, { recursive: false });
  await fs.writeFile(
    path.join(postDir, "meta.ts"),
    `import type { BlogPostMeta } from "../../types";\n\nconst meta = {\n  title: ${JSON.stringify(title)},\n  summary: "Replace with a one-sentence summary.",\n  date: "${today()}",\n  sourceType: "mdx",\n  image: "/assets/visuals/notes-field.png",\n  category: "Build logs",\n  language: "en",\n  source: {\n    status: "original"\n  },\n  tags: ["content system"]\n} satisfies BlogPostMeta;\n\nexport default meta;\n`,
    "utf8"
  );
  await fs.writeFile(
    path.join(postDir, "index.mdx"),
    `## Context\n\nStart the article here.\n\n## Notes\n\n- Add the main points.\n`,
    "utf8"
  );

  console.log(`Created content/blog/posts/${slug}/`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
