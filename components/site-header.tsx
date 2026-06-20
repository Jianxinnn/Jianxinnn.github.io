"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Mail, Search, Share2 } from "lucide-react";
import { BrandMark } from "@/components/brand-mark";
import { profile } from "@/content/profile";

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="site-header">
      <div className="topbar">
        <Link aria-label="Home" className="brand-link" href="/">
          <BrandMark className="brand-logo" />
          <span className="brand-wordmark">
            <span className="brand-name">{profile.siteTitle}</span>
            <span className="brand-kicker">{profile.headerTagline}</span>
          </span>
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
