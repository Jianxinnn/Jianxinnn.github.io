import { viewCountPath } from "@/lib/view-count";

type ViewCountBadgeProps = {
  className?: string;
  scope: string;
  slug: string;
};

export function ViewCountBadge({ className = "", scope, slug }: ViewCountBadgeProps) {
  const counterPath = viewCountPath(scope, slug);
  const params = new URLSearchParams({
    color: "eeeeee",
    label: "views",
    labelColor: "eeeeee",
    style: "flat"
  });

  return (
    <span
      aria-label="View count"
      className={className ? `view-count-badge ${className}` : "view-count-badge"}
      title="Views"
    >
      <img
        alt="views"
        height="18"
        loading="lazy"
        referrerPolicy="no-referrer"
        src={`https://hits.sh/${counterPath}.svg?${params.toString()}`}
      />
    </span>
  );
}

