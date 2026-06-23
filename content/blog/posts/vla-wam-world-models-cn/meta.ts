import type { BlogPostMeta } from "../../types";

const meta = {
  title: "VLA、世界模型与 WAM：机器人模型到底在学什么",
  summary:
    "从学习目标区分 VLA、世界模型与 WAM：VLA 直接生成动作，世界模型预测动作后果，WAM 将动作和未来状态联合建模。",
  date: "2026-06-23",
  sourceType: "mdx",
  image: "/assets/blog/vla-wam-world-models-cn/wam-concept.jpg",
  category: "Technical explainers",
  language: "zh",
  source: {
    status: "original",
    note: "综合 arXiv、PMLR、OpenReview、NVIDIA Research 等公开论文与项目资料写成。"
  },
  tags: ["world models", "physical AI", "reinforcement learning"]
} satisfies BlogPostMeta;

export default meta;
