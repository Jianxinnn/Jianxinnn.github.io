import type { BlogPostMeta } from "../../types";

const meta = {
  title: "AdaWorld: 把 action 写进 world model 预训练",
  summary:
    "一则短 note：AdaWorld 如何从无标注视频中抽取 latent action，并把世界模型适配问题改写成动作接口对齐问题。",
  date: "2026-06-21",
  sourceType: "mdx",
  image: "/assets/blog/adaworld-latent-actions/fig1-framework.jpg",
  category: "Technical explainers",
  language: "zh",
  source: {
    status: "original",
    note: "Reading note on arXiv:2503.18938."
  },
  tags: ["world models", "reinforcement learning"],
  listed: false
} satisfies BlogPostMeta;

export default meta;
