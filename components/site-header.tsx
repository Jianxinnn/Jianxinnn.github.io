"use client";

import { Moon, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { BrandMark } from "@/components/brand-mark";
import { profile } from "@/content/profile";

type Theme = "light" | "dark";

export function SiteHeader() {
  const pathname = usePathname();
  const [theme, setTheme] = useState<Theme>("light");

  useEffect(() => {
    const activeTheme =
      document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    setTheme(activeTheme);
  }, []);

  function toggleTheme() {
    const nextTheme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = nextTheme;
    window.localStorage.setItem("theme", nextTheme);
    setTheme(nextTheme);
  }

  return (
    <header className="site-header">
      <div className="topbar">
        <Link aria-label="Home" className="brand-link" href="/">
          <BrandMark className="brand-logo" />
        </Link>
        <div className="header-actions">
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
          <button
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            className="theme-toggle"
            onClick={toggleTheme}
            title={theme === "dark" ? "Light mode" : "Dark mode"}
            type="button"
          >
            {theme === "dark" ? (
              <Sun aria-hidden="true" size={18} strokeWidth={2.2} />
            ) : (
              <Moon aria-hidden="true" size={18} strokeWidth={2.2} />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
