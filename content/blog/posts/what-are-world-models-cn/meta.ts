import type { BlogPostMeta } from "../../types";

const meta = {
  title: "什么是世界模型",
  summary:
    "从 Ha 与 Schmidhuber 的 V-M-C 架构，到 Genie、V-JEPA、PAN 和 NVIDIA 的物理 AI 视角，梳理世界模型的定义、公式、路线和边界。",
  date: "2026-06-21",
  sourceType: "mdx",
  image: "/assets/blog/world-models-cn/three-paradigms.jpg",
  category: "Technical explainers",
  language: "zh",
  source: {
    status: "original",
    note: "综合本地课件、Edmund Goodman 讲义和 NVIDIA Glossary 写成。"
  },
  tags: ["world models", "physical AI", "reinforcement learning", "illustrated note"]
} satisfies BlogPostMeta;

export default meta;
