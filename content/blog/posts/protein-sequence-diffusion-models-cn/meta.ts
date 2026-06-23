import type { BlogPostMeta } from "../../types"

const meta = {
  title: "蛋白序列扩散模型详解",
  summary:
    "从 token、PLM latent、结构 token 到进化编辑，梳理蛋白序列扩散模型的噪声空间、关键公式和代表论文。",
  date: "2026-06-23",
  sourceType: "mdx",
  image: "/assets/blog/protein-sequence-diffusion-models-cn/fig1-taxonomy.svg",
  category: "Technical explainers",
  language: "zh",
  source: {
    status: "original",
    note: "基于截至 2026-06-23 的本地调研报告，并核对 EvoDiff、DPLM、DPLM-2、DiMA、DSM、MeMDLM、DPLM-Evo 等一手论文和项目页面写成。"
  },
  tags: ["protein design", "diffusion models", "protein language models"]
} satisfies BlogPostMeta

export default meta
