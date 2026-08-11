import { BilingualArticleClient } from "@/components/bilingual-article-client";

export type { BilingualSegment } from "@/components/bilingual-article-content";

type BilingualArticleProps = {
  className?: string;
  initialHtml: string;
  segmentsPath: string;
};

export function BilingualArticle({
  className,
  initialHtml,
  segmentsPath
}: BilingualArticleProps) {
  return (
    <BilingualArticleClient className={className} segmentsPath={segmentsPath}>
      <div dangerouslySetInnerHTML={{ __html: initialHtml }} />
    </BilingualArticleClient>
  );
}
