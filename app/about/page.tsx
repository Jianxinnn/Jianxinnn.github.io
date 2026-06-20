import { SubscribeForm } from "@/components/subscribe-form";
import { profile } from "@/content/profile";

export const metadata = {
  title: "About"
};

export default function AboutPage() {
  return (
    <article className="article-page">
      <h1>About</h1>
      <p>{profile.about}</p>
      <SubscribeForm />
      {profile.aboutSections.map((section) => (
        <section className="article-section" key={section.title}>
          <h2>{section.title}</h2>
          <p>{section.body}</p>
        </section>
      ))}
    </article>
  );
}
