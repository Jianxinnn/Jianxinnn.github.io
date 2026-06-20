import { profile } from "@/content/profile";
import { BrandMark } from "@/components/brand-mark";
import { SubscribeForm } from "@/components/subscribe-form";

export function ProfileSidebar() {
  return (
    <aside className="profile-sidebar" aria-label="Profile summary">
      <div className="profile-sidebar-header">
        <BrandMark className="sidebar-logo" />
        <div>
          <h2>{profile.name}</h2>
          <p className="sidebar-role">{profile.shortRole}</p>
        </div>
      </div>
      <img
        alt={`${profile.name} avatar`}
        className="sidebar-avatar"
        height="96"
        src={profile.assets.avatar}
        width="96"
      />
      <p>{profile.intro}</p>
      <SubscribeForm compact />
      <div className="link-stack">
        {profile.links.map((link) => (
          <a href={link.href} key={link.label}>
            {link.label}
          </a>
        ))}
      </div>
    </aside>
  );
}
