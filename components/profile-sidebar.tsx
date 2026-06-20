import { profile } from "@/content/profile";
import { SubscribeForm } from "@/components/subscribe-form";

export function ProfileSidebar() {
  return (
    <aside className="profile-sidebar" aria-label="Profile summary">
      <div className="sidebar-mark">
        <span className="mark-dot mark-dot-one" />
        <span className="mark-dot mark-dot-two" />
        <span className="mark-dot mark-dot-three" />
      </div>
      <h2>{profile.name}</h2>
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
