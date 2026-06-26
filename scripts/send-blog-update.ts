import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { blogPosts } from "../.generated/blog-posts";
import { profile } from "../content/profile";
import type { BlogPost } from "../content/blog/types";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const recordsPath = path.join(repoRoot, "data/mailings/blog-post-notifications.json");
const buttondownEmailsUrl = "https://api.buttondown.com/v1/emails";

type NotificationRecord = {
  contentSha256: string;
  notifiedAt: string;
  postTitle: string;
  provider: "buttondown" | "manual";
  providerMessageId?: string;
  status: string;
};

type NotificationRecords = Record<string, NotificationRecord>;

function hasArg(name: string) {
  return process.argv.includes(`--${name}`);
}

function siteUrl() {
  return (process.env.SITE_URL || profile.newsletter.siteUrl).replace(/\/+$/, "");
}

function postUrl(post: BlogPost) {
  if (post.href.startsWith("http")) {
    return post.href;
  }

  return `${siteUrl()}${post.href}`;
}

function isNotifiablePost(post: BlogPost) {
  return (
    post.listed !== false &&
    post.sourceType !== "encrypted" &&
    !post.href.startsWith("/protected/")
  );
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", {
    month: "long",
    day: "numeric",
    year: "numeric"
  }).format(new Date(value));
}

function renderEmailBody(post: BlogPost) {
  const tags = post.tags?.length ? post.tags.join(", ") : "";
  const lines = [
    "# New post",
    "",
    `## ${post.title}`,
    "",
    post.summary,
    "",
    `[Read the post](${postUrl(post)})`,
    "",
    `Published: ${formatDate(post.date)}`,
    `Reading time: ${post.readingTime}`
  ];

  if (post.category) {
    lines.push(`Category: ${post.category}`);
  }
  if (tags) {
    lines.push(`Tags: ${tags}`);
  }

  lines.push(
    "",
    "---",
    "",
    `You are receiving this because you subscribed to ${profile.name} updates.`,
    "Subscriber addresses and unsubscribe preferences are managed by Buttondown."
  );

  return `${lines.join("\n")}\n`;
}

function contentHash(post: BlogPost, body: string) {
  return createHash("sha256")
    .update(JSON.stringify({ slug: post.slug, title: post.title, href: post.href, body }))
    .digest("hex");
}

async function readRecords(): Promise<NotificationRecords> {
  try {
    return JSON.parse(await fs.readFile(recordsPath, "utf8")) as NotificationRecords;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return {};
    }
    throw error;
  }
}

async function writeRecords(records: NotificationRecords) {
  await fs.mkdir(path.dirname(recordsPath), { recursive: true });
  await fs.writeFile(recordsPath, `${JSON.stringify(records, null, 2)}\n`, "utf8");
}

async function createButtondownEmail(post: BlogPost, body: string, draft: boolean) {
  const apiKey = process.env.BUTTONDOWN_API_KEY?.trim();

  if (!apiKey) {
    throw new Error("BUTTONDOWN_API_KEY is required to send blog update emails.");
  }

  const payload = {
    subject: `New post: ${post.title}`,
    body,
    status: draft ? "draft" : "about_to_send",
    email_type: "public",
    canonical_url: postUrl(post),
    metadata: {
      project: "personal-profile",
      post_slug: post.slug,
      post_url: postUrl(post)
    }
  };
  const response = await fetch(buttondownEmailsUrl, {
    method: "POST",
    headers: {
      Authorization: `Token ${apiKey}`,
      "Content-Type": "application/json",
      "User-Agent": "personal-profile/1.0",
      ...(draft ? {} : { "X-Buttondown-Live-Dangerously": "true" })
    },
    body: JSON.stringify(payload)
  });
  const responseBody = await response.text();

  if (!response.ok) {
    throw new Error(`Buttondown API error ${response.status}: ${responseBody}`);
  }

  return responseBody.trim() ? (JSON.parse(responseBody) as Record<string, unknown>) : {};
}

async function main() {
  const send = hasArg("send");
  const draft = hasArg("draft");
  const records = await readRecords();
  const postsToNotify = blogPosts
    .filter(isNotifiablePost)
    .filter((post) => !records[post.slug])
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  if (postsToNotify.length === 0) {
    console.log("No new public blog posts to notify.");
    return;
  }

  for (const post of postsToNotify) {
    const body = renderEmailBody(post);
    const hash = contentHash(post, body);

    if (!send) {
      console.log(`Dry run: would notify ${post.slug}`);
      console.log(`Subject: New post: ${post.title}`);
      console.log(`URL: ${postUrl(post)}`);
      console.log(`Content SHA-256: ${hash}`);
      continue;
    }

    const result = await createButtondownEmail(post, body, draft);
    const status = String(result.status || (draft ? "draft" : "about_to_send"));
    const providerMessageId = String(result.id || result.slug || "");

    records[post.slug] = {
      contentSha256: hash,
      notifiedAt: new Date().toISOString(),
      postTitle: post.title,
      provider: "buttondown",
      ...(providerMessageId ? { providerMessageId } : {}),
      status
    };
    await writeRecords(records);
    console.log(`${draft ? "Created draft" : "Queued email"} for ${post.slug}: ${status}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
