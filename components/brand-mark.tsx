import { profile } from "@/content/profile";

type BrandMarkProps = {
  className?: string;
};

export function BrandMark({ className = "" }: BrandMarkProps) {
  return (
    <img
      alt=""
      aria-hidden="true"
      className={className ? `brand-mark ${className}` : "brand-mark"}
      height="40"
      src={profile.assets.logo}
      width="40"
    />
  );
}
