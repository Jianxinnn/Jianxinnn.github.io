import AdaworldLatentActions from "@/content/notes/adaworld-latent-actions/index.mdx";
import PomdpBeliefStateControl from "@/content/notes/pomdp-belief-state-control/index.mdx";
import type { Note } from "@/content/notes/types";

export const notes = [
  {
    slug: "pomdp-belief-state-control",
    title: "POMDP belief-state control",
    summary:
      "POMDP 把状态不可见的序列决策写成 belief state 上的控制问题，适合需要边观察、边更新、边行动的场景。",
    date: "2026-06-25",
    updated: "2026-06-25T16:59:00+08:00",
    readingTime: "3 min read",
    image: "/assets/notes/pomdp-belief-state-control.png",
    tags: ["reinforcement learning", "planning under uncertainty"],
    Content: PomdpBeliefStateControl
  },
  {
    slug: "adaworld-latent-actions",
    title: "AdaWorld latent actions",
    summary:
      "A short note on action-aware pretraining as a reusable control interface for adaptable world models.",
    date: "2026-06-21",
    updated: "2026-06-21T23:59:00+08:00",
    readingTime: "2 min read",
    image: "/assets/blog/adaworld-latent-actions/fig1-framework.jpg",
    tags: ["world models", "reinforcement learning"],
    Content: AdaworldLatentActions
  }
] satisfies Note[];

export function getNote(slug: string) {
  return notes.find((note) => note.slug === slug);
}
