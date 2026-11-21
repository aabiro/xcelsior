"use client";

import { useEffect, useRef, useState, type ComponentProps, type CSSProperties } from "react";
import { ResponsiveContainer as BaseResponsiveContainer } from "recharts";

export * from "recharts";

type ResponsiveContainerProps = ComponentProps<typeof BaseResponsiveContainer>;

function hasConcreteHeight(
  rectHeight: number,
  height: ResponsiveContainerProps["height"],
  minHeight: ResponsiveContainerProps["minHeight"],
  aspect: ResponsiveContainerProps["aspect"],
) {
  if (rectHeight > 0) return true;
  if (typeof height === "number") return height > 0;
  if (typeof height === "string" && height !== "100%") return true;
  if (typeof minHeight === "number") return minHeight > 0;
  if (typeof minHeight === "string" && minHeight.trim().length > 0) return true;
  return typeof aspect === "number" && aspect > 0;
}

function resolveInitialHeight(
  rectHeight: number,
  rectWidth: number,
  height: ResponsiveContainerProps["height"],
  minHeight: ResponsiveContainerProps["minHeight"],
  aspect: ResponsiveContainerProps["aspect"],
): number {
  if (rectHeight > 0) return Math.round(rectHeight);
  if (typeof height === "number" && height > 0) return height;
  if (typeof minHeight === "number" && minHeight > 0) return minHeight;
  if (typeof aspect === "number" && aspect > 0 && rectWidth > 0) {
    return Math.max(1, Math.round(rectWidth / aspect));
  }
  return 0;
}

export function ResponsiveContainer({
  children,
  className,
  style,
  minWidth,
  minHeight,
  maxHeight,
  width,
  height,
  aspect,
  initialDimension,
  ...props
}: ResponsiveContainerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [measuredSize, setMeasuredSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const node = hostRef.current;
    if (!node) return;

    const update = () => {
      const rect = node.getBoundingClientRect();
      if (rect.width <= 0 || !hasConcreteHeight(rect.height, height, minHeight, aspect)) {
        setMeasuredSize(null);
        return;
      }
      const nextWidth = Math.max(1, Math.round(rect.width));
      const resolvedHeight = resolveInitialHeight(rect.height, nextWidth, height, minHeight, aspect);
      const nextHeight = Math.max(1, resolvedHeight || nextWidth);
      setMeasuredSize((prev) =>
        prev && prev.width === nextWidth && prev.height === nextHeight
          ? prev
          : { width: nextWidth, height: nextHeight },
      );
    };

    update();

    if (typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(() => update());
    observer.observe(node);

    return () => observer.disconnect();
  }, [aspect, height, minHeight]);

  const wrapperStyle: CSSProperties = {
    width: width ?? "100%",
    height,
    minWidth: minWidth ?? 0,
    minHeight,
    maxHeight,
    ...style,
  };
  const wrapperClassName = typeof className === "string" ? className : undefined;

  return (
    <div ref={hostRef} className={wrapperClassName} style={wrapperStyle}>
      {measuredSize ? (
        <BaseResponsiveContainer
          {...props}
          width={width}
          height={height}
          minWidth={minWidth ?? 0}
          minHeight={minHeight}
          maxHeight={maxHeight}
          aspect={aspect}
          initialDimension={initialDimension ?? measuredSize}
        >
          {children}
        </BaseResponsiveContainer>
      ) : null}
    </div>
  );
}
