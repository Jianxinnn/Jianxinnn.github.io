"use client";

import { type FormEvent, useState } from "react";
import { Lock, RotateCcw, Unlock } from "lucide-react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";

type ProtectedArticlePost = {
  slug: string;
  title: string;
  encryptedPath: string;
};

type EncryptedArticleFile = {
  version: 1;
  algorithm: "AES-GCM";
  kdf: "PBKDF2-SHA-256";
  iterations: number;
  salt: string;
  iv: string;
  ciphertext: string;
};

type DecryptedArticle = {
  version: 1;
  slug: string;
  title: string;
  summary: string;
  date: string;
  markdown: string;
};

type ProtectedArticleProps = {
  post: ProtectedArticlePost;
};

const decoder = new TextDecoder();
const encoder = new TextEncoder();

function base64ToBytes(value: string) {
  const binary = window.atob(value);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

function isEncryptedArticleFile(value: unknown): value is EncryptedArticleFile {
  const candidate = value as Partial<EncryptedArticleFile>;

  return (
    candidate.version === 1 &&
    candidate.algorithm === "AES-GCM" &&
    candidate.kdf === "PBKDF2-SHA-256" &&
    typeof candidate.iterations === "number" &&
    typeof candidate.salt === "string" &&
    typeof candidate.iv === "string" &&
    typeof candidate.ciphertext === "string"
  );
}

function isDecryptedArticle(value: unknown): value is DecryptedArticle {
  const candidate = value as Partial<DecryptedArticle>;

  return (
    candidate.version === 1 &&
    typeof candidate.slug === "string" &&
    typeof candidate.title === "string" &&
    typeof candidate.summary === "string" &&
    typeof candidate.date === "string" &&
    typeof candidate.markdown === "string"
  );
}

async function deriveKey(password: string, encrypted: EncryptedArticleFile) {
  const keyMaterial = await window.crypto.subtle.importKey(
    "raw",
    encoder.encode(password),
    "PBKDF2",
    false,
    ["deriveKey"]
  );

  return window.crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      hash: "SHA-256",
      salt: base64ToBytes(encrypted.salt),
      iterations: encrypted.iterations
    },
    keyMaterial,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"]
  );
}

async function decryptArticle(password: string, encryptedPath: string) {
  const response = await fetch(encryptedPath, { cache: "no-store" });

  if (!response.ok) {
    throw new Error("Encrypted article file was not found.");
  }

  const encrypted = await response.json();

  if (!isEncryptedArticleFile(encrypted)) {
    throw new Error("Encrypted article file is invalid.");
  }

  const key = await deriveKey(password, encrypted);
  const plaintext = await window.crypto.subtle.decrypt(
    {
      name: "AES-GCM",
      iv: base64ToBytes(encrypted.iv)
    },
    key,
    base64ToBytes(encrypted.ciphertext)
  );
  const decrypted = JSON.parse(decoder.decode(plaintext));

  if (!isDecryptedArticle(decrypted)) {
    throw new Error("Decrypted article payload is invalid.");
  }

  return decrypted;
}

function protectedUrlTransform(url: string, key: string) {
  if (key === "src" && /^data:image\/(?:png|jpe?g|webp|gif);base64,/i.test(url)) {
    return url;
  }

  return defaultUrlTransform(url);
}

export function ProtectedArticle({ post }: ProtectedArticleProps) {
  const [password, setPassword] = useState("");
  const [article, setArticle] = useState<DecryptedArticle | null>(null);
  const [error, setError] = useState("");
  const [isUnlocking, setIsUnlocking] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsUnlocking(true);

    try {
      const decrypted = await decryptArticle(password, post.encryptedPath);

      if (decrypted.slug !== post.slug) {
        throw new Error("Encrypted article slug does not match this page.");
      }

      setArticle(decrypted);
      setPassword("");
    } catch {
      setArticle(null);
      setError("Password did not unlock this article.");
    } finally {
      setIsUnlocking(false);
    }
  }

  if (article) {
    return (
      <section className="protected-article-unlocked">
        <div className="protected-article-toolbar" data-pagefind-ignore>
          <span>Unlocked in this browser tab.</span>
          <button type="button" onClick={() => setArticle(null)}>
            <Lock size={16} strokeWidth={2} />
            Lock
          </button>
        </div>
        <div className="mdx-body protected-article-body">
          <ReactMarkdown urlTransform={protectedUrlTransform}>
            {article.markdown}
          </ReactMarkdown>
        </div>
      </section>
    );
  }

  return (
    <section className="protected-article-card" data-pagefind-ignore>
      <div className="protected-article-card-icon" aria-hidden="true">
        <Lock size={20} strokeWidth={2} />
      </div>
      <div className="protected-article-card-copy">
        <h2>Protected article</h2>
        <p>Enter the article password to decrypt the text locally in this browser.</p>
      </div>
      <form className="protected-password-form" onSubmit={handleSubmit}>
        <label htmlFor="protected-article-password">Password</label>
        <div className="protected-password-row">
          <input
            autoComplete="off"
            id="protected-article-password"
            onChange={(event) => setPassword(event.target.value)}
            required
            type="password"
            value={password}
          />
          <button disabled={isUnlocking} type="submit">
            {isUnlocking ? (
              <RotateCcw size={16} strokeWidth={2} />
            ) : (
              <Unlock size={16} strokeWidth={2} />
            )}
            {isUnlocking ? "Unlocking" : "Unlock"}
          </button>
        </div>
        {error ? <p className="protected-article-error">{error}</p> : null}
      </form>
    </section>
  );
}
