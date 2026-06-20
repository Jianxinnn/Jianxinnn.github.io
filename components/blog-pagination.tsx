import Link from "next/link";

type BlogPaginationProps = {
  currentPage: number;
  totalPages: number;
};

function pageHref(page: number) {
  return page === 1 ? "/blog" : `/blog/p/${page}`;
}

export function BlogPagination({ currentPage, totalPages }: BlogPaginationProps) {
  if (totalPages <= 1) {
    return null;
  }

  return (
    <nav aria-label="Blog pagination" className="blog-pagination">
      {currentPage > 1 ? (
        <Link href={pageHref(currentPage - 1)}>Newer</Link>
      ) : (
        <span aria-hidden="true">Newer</span>
      )}
      <div>
        {Array.from({ length: totalPages }, (_, index) => index + 1).map((page) => (
          <Link
            aria-current={page === currentPage ? "page" : undefined}
            href={pageHref(page)}
            key={page}
          >
            {page}
          </Link>
        ))}
      </div>
      {currentPage < totalPages ? (
        <Link href={pageHref(currentPage + 1)}>Older</Link>
      ) : (
        <span aria-hidden="true">Older</span>
      )}
    </nav>
  );
}
