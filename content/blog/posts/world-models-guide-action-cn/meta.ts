import type { BlogPostMeta } from "../../types";

const meta = {
  title: "世界模型如何指导行动",
  summary:
    "以动作闭环为主线，梳理世界模型从 model-based RL、latent dynamics 到交互式视频模型和机器人行动接口的发展脉络。",
  date: "2026-06-23",
  sourceType: "mdx",
  image: "/assets/blog/world-models-guide-action-cn/pan-latent-world-model.jpg",
  category: "Technical explainers",
  language: "zh",
  source: {
    status: "original",
    note: "基于 arXiv、OpenReview、ACM、NeurIPS、PMLR 等公开论文与项目资料写成。"
  },
  tags: ["world models", "reinforcement learning", "physical AI"]
} satisfies BlogPostMeta;

export default meta;
