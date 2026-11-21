import { describe, expect, it, vi, afterEach } from "vitest";
import { render } from "@testing-library/react";
import { ResponsiveContainer, BarChart, Bar } from "@/lib/recharts";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ResponsiveContainer wrapper", () => {
  it("forwards sizing styles to the host wrapper and seeds initialDimension so Recharts never warns width(-1)/height(-1)", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 640,
      height: 224,
      top: 0,
      left: 0,
      right: 640,
      bottom: 224,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    const { container } = render(
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={180} minWidth={0} debounce={1}>
          <BarChart data={[{ date: "Nov 21", amount: 12 }]}>
            <Bar dataKey="amount" />
          </BarChart>
        </ResponsiveContainer>
      </div>,
    );

    const host = container.querySelector(".h-56 > div") as HTMLElement | null;
    expect(host).not.toBeNull();
    expect(host?.style.width).toBe("100%");
    expect(host?.style.height).toBe("100%");
    expect(host?.style.minHeight).toBe("180px");
    expect(container.querySelector(".recharts-responsive-container")).not.toBeNull();

    const negativeDimWarnings = warnSpy.mock.calls.filter(([msg]) =>
      String(msg).includes("should be greater than 0"),
    );
    expect(negativeDimWarnings).toHaveLength(0);
  });
});
