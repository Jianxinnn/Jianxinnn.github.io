import type { BlogPostMeta } from "../../types";

const meta = {
  title: "扩散模型的数学导论",
  summary:
    "Jianfeng Lu 关于扩散模型数学基础的中文网页稿，按采样视角串联朗之万动力学、score-based diffusion、离散化误差、离散扩散和推理时控制。",
  date: "2026-07-06",
  readingTime: "63 pages",
  sourceType: "html",
  href: "/blog/diffusion-models-math-intro-cn/",
  category: "Technical explainers",
  language: "zh",
  source: {
    status: "translation",
    label: "转载 / 翻译",
    originalTitle: "A Mathematical Introduction to Diffusion Models",
    originalUrl: "https://arxiv.org/abs/2607.01693"
  },
  tags: ["diffusion models", "flow matching"]
} satisfies BlogPostMeta;

export default meta;
