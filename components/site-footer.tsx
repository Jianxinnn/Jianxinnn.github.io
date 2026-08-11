import Link from "next/link";
import { profile } from "@/content/profile";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <p>© 2026 {profile.name}</p>
        <div className="footer-links">
          <Link href="/about" prefetch={false}>About</Link>
          <Link href="/blog" prefetch={false}>Blog</Link>
          <Link href="/notes" prefetch={false}>Notes</Link>
          <Link href="/readings" prefetch={false}>Readings</Link>
          <a href={`mailto:${profile.email}`}>Contact</a>
        </div>
      </div>
    </footer>
  );
}
