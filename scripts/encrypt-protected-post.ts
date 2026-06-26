import { randomBytes, webcrypto } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { ProtectedPostRecord } from "../content/protected-posts";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const privatePostsRoot = path.join(repoRoot, "private/protected-posts");
const passwordRegistryPath = path.join(repoRoot, "private/protected-post-passwords.md");
const metadataPath = path.join(repoRoot, "content/protected-posts.json");
const encoder = new TextEncoder();

type Args = Record<string, string | boolean>;

type EncryptedArticle = {
  version: 1;
  algorithm: "AES-GCM";
  kdf: "PBKDF2-SHA-256";
  iterations: number;
  salt: string;
  iv: string;
  ciphertext: string;
  encryptedAt: string;
};

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

function toBase64(value: ArrayBuffer | Uint8Array) {
  return Buffer.from(value instanceof Uint8Array ? value : new Uint8Array(value)).toString(
    "base64"
  );
}

function parseIterations(value: string) {
  if (!value) {
    return 800_000;
  }

  const parsed = Number(value);

  if (!Number.isInteger(parsed) || parsed < 100_000) {
    throw new Error("Iterations must be an integer >= 100000.");
  }

  return parsed;
}

function parsePasswordRegistry(source: string) {
  const passwords = new Map<string, string>();

  for (const line of source.split("\n")) {
    const trimmed = line.trim();

    if (!trimmed.startsWith("|") || trimmed.includes("---")) {
      continue;
    }

    const cells = trimmed
      .split("|")
      .slice(1, -1)
      .map((cell) => cell.trim());

    if (cells[0] && cells[0] !== "slug" && cells[1]) {
      passwords.set(cells[0], cells[1]);
    }
  }

  return passwords;
}

async function readMetadata() {
  return JSON.parse(await fs.readFile(metadataPath, "utf8")) as ProtectedPostRecord[];
}

function outputPathForPost(post: ProtectedPostRecord) {
  const encryptedPath = post.encryptedPath ?? `/protected/${post.slug}.json`;
  const relativePath = encryptedPath.replace(/^\/+/, "");

  if (relativePath.split(/[\\/]/).includes("..")) {
    throw new Error(`${post.slug}: encryptedPath cannot contain '..'.`);
  }

  return path.join(repoRoot, "public", relativePath);
}

async function encryptPost(
  post: ProtectedPostRecord,
  password: string,
  iterations: number
) {
  const plaintextPath = path.join(privatePostsRoot, `${post.slug}.md`);
  const markdown = await fs.readFile(plaintextPath, "utf8");
  const salt = randomBytes(16);
  const iv = randomBytes(12);
  const keyMaterial = await webcrypto.subtle.importKey(
    "raw",
    encoder.encode(password),
    "PBKDF2",
    false,
    ["deriveKey"]
  );
  const key = await webcrypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      hash: "SHA-256",
      salt,
      iterations
    },
    keyMaterial,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt"]
  );
  const payload = JSON.stringify({
    version: 1,
    slug: post.slug,
    title: post.title,
    summary: post.summary,
    date: post.date,
    markdown
  });
  const ciphertext = await webcrypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    encoder.encode(payload)
  );
  const encrypted: EncryptedArticle = {
    version: 1,
    algorithm: "AES-GCM",
    kdf: "PBKDF2-SHA-256",
    iterations,
    salt: toBase64(salt),
    iv: toBase64(iv),
    ciphertext: toBase64(ciphertext),
    encryptedAt: new Date().toISOString()
  };
  const outputPath = outputPathForPost(post);

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(encrypted, null, 2)}\n`, "utf8");

  return outputPath;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const slug = getStringArg(args, "slug");
  const iterations = parseIterations(getStringArg(args, "iterations"));
  const encryptAll = args.all === true;

  if (!slug && !encryptAll) {
    throw new Error("Usage: npm run protected:encrypt -- --slug my-post");
  }

  const records = await readMetadata();
  const passwordRegistry = parsePasswordRegistry(
    await fs.readFile(passwordRegistryPath, "utf8")
  );
  const posts = encryptAll ? records : records.filter((post) => post.slug === slug);

  if (posts.length === 0) {
    throw new Error(`No protected post metadata found for slug: ${slug}`);
  }

  for (const post of posts) {
    const password = passwordRegistry.get(post.slug);

    if (!password) {
      throw new Error(
        `${post.slug}: add a password row to private/protected-post-passwords.md`
      );
    }

    const outputPath = await encryptPost(post, password, iterations);
    console.log(`Encrypted ${post.slug} -> ${path.relative(repoRoot, outputPath)}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
