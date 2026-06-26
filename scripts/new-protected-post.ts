import { randomBytes } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { ProtectedPostRecord } from "../content/protected-posts";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const privateRoot = path.join(repoRoot, "private");
const privatePostsRoot = path.join(privateRoot, "protected-posts");
const passwordRegistryPath = path.join(privateRoot, "protected-post-passwords.md");
const metadataPath = path.join(repoRoot, "content/protected-posts.json");

type Args = Record<string, string | boolean>;

function parseArgs(argv: string[]) {
  const args: Args = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (!arg.startsWith("--")) {
      continue;
    }

    const key = arg.slice(2);
    const next = argv[index + 1];

    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      index += 1;
    }
  }

  return args;
}

function getStringArg(args: Args, key: string) {
  const value = args[key];
  return typeof value === "string" ? value.trim() : "";
}

function localDate() {
  const now = new Date();
  const localTime = now.getTime() - now.getTimezoneOffset() * 60_000;
  return new Date(localTime).toISOString().slice(0, 10);
}

function validateSlug(slug: string) {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
    throw new Error("Slug must use lowercase letters, numbers, and hyphens.");
  }
}

function splitTags(value: string) {
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

async function ensurePasswordRegistry() {
  await fs.mkdir(privateRoot, { recursive: true });

  try {
    await fs.access(passwordRegistryPath);
  } catch {
    await fs.writeFile(
      passwordRegistryPath,
      [
        "# Protected post passwords",
        "",
        "This file is ignored by Git. Keep it local and back it up in a password manager.",
        "",
        "| slug | password | updated | plaintext |",
        "| --- | --- | --- | --- |",
        ""
      ].join("\n"),
      "utf8"
    );
  }
}

async function readMetadata() {
  try {
    return JSON.parse(await fs.readFile(metadataPath, "utf8")) as ProtectedPostRecord[];
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }

    throw error;
  }
}

async function writeMetadata(records: ProtectedPostRecord[]) {
  await fs.writeFile(metadataPath, `${JSON.stringify(records, null, 2)}\n`, "utf8");
}

async function appendPassword(slug: string, plaintextPath: string) {
  await ensurePasswordRegistry();
  const registry = await fs.readFile(passwordRegistryPath, "utf8");

  if (registry.split("\n").some((line) => line.startsWith(`| ${slug} |`))) {
    return false;
  }

  const password = randomBytes(24).toString("base64url");
  const relativePlaintext = path.relative(repoRoot, plaintextPath);
  const row = `| ${slug} | ${password} | ${localDate()} | ${relativePlaintext} |`;

  await fs.writeFile(
    passwordRegistryPath,
    registry.endsWith("\n") ? `${registry}${row}\n` : `${registry}\n${row}\n`,
    "utf8"
  );

  return true;
}

async function createPlaintext(slug: string, title: string) {
  await fs.mkdir(privatePostsRoot, { recursive: true });
  const plaintextPath = path.join(privatePostsRoot, `${slug}.md`);

  try {
    await fs.access(plaintextPath);
    return { created: false, plaintextPath };
  } catch {
    await fs.writeFile(
      plaintextPath,
      [
        `# ${title}`,
        "",
        "Write the protected article here.",
        "",
        "Keep sensitive assets inside this Markdown file or encrypt them separately.",
        ""
      ].join("\n"),
      "utf8"
    );

    return { created: true, plaintextPath };
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const slug = getStringArg(args, "slug");
  const title = getStringArg(args, "title");
  const summary = getStringArg(args, "summary");
  const date = getStringArg(args, "date") || localDate();
  const category = getStringArg(args, "category");
  const tags = splitTags(getStringArg(args, "tags"));

  if (!slug || !title || !summary) {
    throw new Error(
      'Usage: npm run protected:new -- --slug my-post --title "My Post" --summary "Short public summary"'
    );
  }

  validateSlug(slug);

  const { created, plaintextPath } = await createPlaintext(slug, title);
  const passwordAdded = await appendPassword(slug, plaintextPath);
  const records = await readMetadata();
  const existing = records.find((record) => record.slug === slug);

  if (!existing) {
    records.push({
      slug,
      title,
      summary,
      date,
      readingTime: "Protected",
      ...(category ? { category } : {}),
      ...(tags.length ? { tags } : {})
    });
    await writeMetadata(records);
  }

  console.log(`${created ? "Created" : "Found"} ${path.relative(repoRoot, plaintextPath)}`);
  console.log(`${passwordAdded ? "Added" : "Found"} password row in private/protected-post-passwords.md`);
  console.log(`${existing ? "Found" : "Added"} metadata in content/protected-posts.json`);
  console.log("Run npm run protected:encrypt -- --slug " + slug + " after editing the plaintext.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
