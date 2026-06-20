import type { ComponentType } from "react";
import AgenticResearchWorkbench from "@/content/blog/agentic-research-workbench.mdx";
import PublicationProfileDesign from "@/content/blog/publication-profile-design.mdx";

export const blogPostContent: Record<string, ComponentType> = {
  "agentic-research-workbench": AgenticResearchWorkbench,
  "publication-profile-design": PublicationProfileDesign
};
