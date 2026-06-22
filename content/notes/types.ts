import type { ComponentType } from "react";

export type Note = {
  slug: string;
  title: string;
  summary: string;
  date: string;
  readingTime: string;
  Content: ComponentType;
  image?: string;
  tags?: string[];
  updated?: string;
};
