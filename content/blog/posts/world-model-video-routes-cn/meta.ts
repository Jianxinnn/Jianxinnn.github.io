import type { BlogPostMeta } from "../../types";

const meta = {
  title: "Sora、JEPA、Genie、Marble 的技术路线",
  summary:
    "比较 Sora、JEPA、Genie、Marble 的变量、目标和接口，围绕 video latent、表征预测、latent action dynamics 与 3D world state。",
  date: "2026-06-24",
  sourceType: "mdx",
  image: "/assets/blog/world-model-video-routes-cn/fig1-sora.png",
  category: "Technical explainers",
  language: "zh",
  source: {
    status: "original",
    note: "基于 OpenAI、Meta、Google DeepMind、World Labs 官方资料及相关论文整理，信息核对至 2026-06-24。"
  },
  tags: ["world models", "physical AI", "diffusion models"]
} satisfies BlogPostMeta;

export default meta;
