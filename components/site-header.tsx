"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Mail, Search, Share2 } from "lucide-react";
import { profile } from "@/content/profile";

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="site-header">
      <div className="topbar">
        <Link aria-label="Home" className="mark" href="/">
          <span className="mark-dot mark-dot-one" />
          <span className="mark-dot mark-dot-two" />
          <span className="mark-dot mark-dot-three" />
        </Link>
        <Link className="site-title" href="/">
          {profile.siteTitle}
        </Link>
        <div className="header-actions">
          <button aria-label="Search" className="icon-button" title="Search">
            <Search size={22} strokeWidth={2} />
          </button>
          <button aria-label="Share" className="icon-button" title="Share">
            <Share2 size={21} strokeWidth={2} />
          </button>
          <a className="primary-button" href={`mailto:${profile.email}`}>
            <Mail size={17} strokeWidth={2} />
            <span>Contact</span>
          </a>
        </div>
      </div>
      <nav aria-label="Main navigation" className="section-tabs">
        {profile.nav.map((item) => {
          const active =
            item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          return (
            <Link
              aria-current={active ? "page" : undefined}
              className={active ? "tab active" : "tab"}
              href={item.href}
              key={item.href}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
