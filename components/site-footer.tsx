import Link from "next/link";
import { profile } from "@/content/profile";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <p>© 2026 {profile.name}</p>
        <div className="footer-links">
          <Link href="/about">About</Link>
          <Link href="/archive">Archive</Link>
          <a href={`mailto:${profile.email}`}>Contact</a>
        </div>
      </div>
    </footer>
  );
}
