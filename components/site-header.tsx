"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BrandMark } from "@/components/brand-mark";
import { profile } from "@/content/profile";

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="site-header">
      <div className="topbar">
        <Link aria-label="Home" className="brand-link" href="/">
          <BrandMark className="brand-logo" />
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
      </div>
    </header>
  );
}
