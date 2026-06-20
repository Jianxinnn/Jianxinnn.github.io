"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Mail } from "lucide-react";
import { BrandMark } from "@/components/brand-mark";
import { profile } from "@/content/profile";

export function SiteHeader() {
  const pathname = usePathname();
  const scholarLink = profile.links.find((link) => link.label === "Google Scholar");

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
        <div className="header-actions">
          {scholarLink ? (
            <a className="secondary-button" href={scholarLink.href}>
              Scholar
            </a>
          ) : null}
          <a className="primary-button" href={`mailto:${profile.email}`}>
            <Mail size={17} strokeWidth={2} />
            <span>Contact</span>
          </a>
        </div>
      </div>
    </header>
  );
}
