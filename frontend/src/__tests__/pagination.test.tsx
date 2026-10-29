import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { renderHook } from "@testing-library/react";

describe("usePagination", () => {
  it("returns totalPages = 1 for empty array", () => {
    const { result } = renderHook(() => usePagination([], 10));
    expect(result.current.totalPages).toBe(1);
  });

  it("computes totalPages correctly", () => {
    const items = Array.from({ length: 25 }, (_, i) => i);
    const { result } = renderHook(() => usePagination(items, 10));
    expect(result.current.totalPages).toBe(3);
  });

  it("paginates to the correct slice", () => {
    const items = [1, 2, 3, 4, 5, 6, 7];
    const { result } = renderHook(() => usePagination(items, 3));
    expect(result.current.paginate(1)).toEqual([1, 2, 3]);
    expect(result.current.paginate(2)).toEqual([4, 5, 6]);
    expect(result.current.paginate(3)).toEqual([7]);
  });

  it("returns empty slice for out-of-range page", () => {
    const items = [1, 2, 3];
    const { result } = renderHook(() => usePagination(items, 10));
    expect(result.current.paginate(5)).toEqual([]);
  });
});

describe("Pagination component", () => {
  it("does not render when totalPages <= 1", () => {
    const { container } = render(
      <Pagination page={1} totalPages={1} onPageChange={() => {}} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders page buttons", () => {
    render(<Pagination page={1} totalPages={3} onPageChange={() => {}} />);
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("disables previous button on first page", () => {
    render(<Pagination page={1} totalPages={3} onPageChange={() => {}} />);
    expect(screen.getByLabelText("Previous page")).toBeDisabled();
  });

  it("disables next button on last page", () => {
    render(<Pagination page={3} totalPages={3} onPageChange={() => {}} />);
    expect(screen.getByLabelText("Next page")).toBeDisabled();
  });

  it("calls onPageChange when a page button is clicked", () => {
    const onPageChange = vi.fn();
    render(<Pagination page={1} totalPages={3} onPageChange={onPageChange} />);
    fireEvent.click(screen.getByText("2"));
    expect(onPageChange).toHaveBeenCalledWith(2);
  });

  it("shows ellipsis for large page counts", () => {
    render(<Pagination page={5} totalPages={10} onPageChange={() => {}} />);
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getAllByText("…").length).toBeGreaterThanOrEqual(1);
  });
});
