import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { blogPosts } from "../.generated/blog-posts";
import { allowedBlogTags } from "../content/blog/tags";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const allowedTags = new Set<string>(allowedBlogTags);

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function isValidDate(value: string) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(new Date(value).getTime());
}

async function main() {
  const errors: string[] = [];
  const slugs = new Set<string>();

  for (const post of blogPosts) {
    if (slugs.has(post.slug)) {
      errors.push(`${post.slug}: duplicate slug`);
    }
    slugs.add(post.slug);

    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(post.slug)) {
      errors.push(`${post.slug}: slug must use lowercase kebab-case`);
    }

    if (!post.title.trim()) errors.push(`${post.slug}: title is required`);
    if (!post.summary.trim()) errors.push(`${post.slug}: summary is required`);
    if (!isValidDate(post.date)) errors.push(`${post.slug}: date must be YYYY-MM-DD`);
    if (!post.readingTime.trim()) errors.push(`${post.slug}: readingTime is required`);

    if (post.sourceType === "mdx") {
      const mdxPath = path.join(repoRoot, "content/blog/posts", post.slug, "index.mdx");
      if (!(await exists(mdxPath))) {
        errors.push(`${post.slug}: missing ${path.relative(repoRoot, mdxPath)}`);
      }
      if (post.href !== `/blog/${post.slug}/`) {
        errors.push(`${post.slug}: mdx href must be /blog/${post.slug}/`);
      }
    }

    if (post.sourceType === "html") {
      const htmlPath = path.join(repoRoot, "public/blog", post.slug, "index.html");
      if (!(await exists(htmlPath))) {
        errors.push(`${post.slug}: missing ${path.relative(repoRoot, htmlPath)}`);
      }
      if (post.href !== `/blog/${post.slug}/`) {
        errors.push(`${post.slug}: html href must be /blog/${post.slug}/`);
      }
    }

    if (post.sourceType === "external" && !/^https?:\/\//.test(post.href)) {
      errors.push(`${post.slug}: external href must start with http:// or https://`);
    }

    if (post.image?.startsWith("/")) {
      const imagePath = path.join(repoRoot, "public", post.image);
      if (!(await exists(imagePath))) {
        errors.push(`${post.slug}: missing image ${post.image}`);
      }
    }

    for (const tag of post.tags ?? []) {
      if (!allowedTags.has(tag)) {
        errors.push(`${post.slug}: unknown tag "${tag}" in content/blog/tags.ts`);
      }
    }
  }

  if (errors.length) {
    console.error(errors.map((error) => `- ${error}`).join("\n"));
    process.exit(1);
  }

  console.log(`Validated ${blogPosts.length} blog posts.`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
