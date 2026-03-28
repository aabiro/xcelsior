"use client";

import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface PaginationProps {
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  className?: string;
}

export function Pagination({ page, totalPages, onPageChange, className }: PaginationProps) {
  if (totalPages <= 1) return null;

  // Generate page numbers: show first, last, current ± 1, with ellipsis
  const pages: (number | "ellipsis")[] = [];
  for (let i = 1; i <= totalPages; i++) {
    if (i === 1 || i === totalPages || (i >= page - 1 && i <= page + 1)) {
      pages.push(i);
    } else if (pages[pages.length - 1] !== "ellipsis") {
      pages.push("ellipsis");
    }
  }

  return (
    <div className={cn("flex items-center justify-center gap-1", className)}>
      <button
        disabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
        className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:bg-surface-hover hover:text-text-primary disabled:opacity-40 disabled:pointer-events-none transition-colors"
        aria-label="Previous page"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>
      {pages.map((p, i) =>
        p === "ellipsis" ? (
          <span key={`e${i}`} className="flex h-8 w-8 items-center justify-center text-text-muted text-sm">
            …
          </span>
        ) : (
          <button
            key={p}
            onClick={() => onPageChange(p)}
            className={cn(
              "inline-flex h-8 w-8 items-center justify-center rounded-lg text-sm transition-colors",
              p === page
                ? "bg-accent-red text-white font-medium"
                : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
            )}
          >
            {p}
          </button>
        ),
      )}
      <button
        disabled={page >= totalPages}
        onClick={() => onPageChange(page + 1)}
        className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:bg-surface-hover hover:text-text-primary disabled:opacity-40 disabled:pointer-events-none transition-colors"
        aria-label="Next page"
      >
        <ChevronRight className="h-4 w-4" />
      </button>
    </div>
  );
}

/** Hook to paginate an array client-side. */
export function usePagination<T>(items: T[], perPage = 10) {
  const totalPages = Math.max(1, Math.ceil(items.length / perPage));
  return {
    paginate: (page: number) => {
      const start = (page - 1) * perPage;
      return items.slice(start, start + perPage);
    },
    totalPages,
  };
}
