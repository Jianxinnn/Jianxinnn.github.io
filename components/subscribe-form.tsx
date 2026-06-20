import { profile } from "@/content/profile";

type SubscribeFormProps = {
  compact?: boolean;
};

export function SubscribeForm({ compact = false }: SubscribeFormProps) {
  return (
    <form action={`mailto:${profile.email}`} className={compact ? "signup compact" : "signup"}>
      <input
        aria-label="Email address"
        name="email"
        placeholder="Type your email..."
        type="email"
      />
      <button type="submit">Follow</button>
    </form>
  );
}
