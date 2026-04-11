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

export function ResponsiveContainer({
  children,
  className,
  style,
  minWidth,
  minHeight,
  width,
  height,
  aspect,
  ...props
}: ResponsiveContainerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const node = hostRef.current;
    if (!node) return;

    const update = () => {
      const rect = node.getBoundingClientRect();
      setReady(rect.width > 0 && hasConcreteHeight(rect.height, height, minHeight, aspect));
    };

    update();

    const observer = new ResizeObserver(() => update());
    observer.observe(node);

    return () => observer.disconnect();
  }, [aspect, height, minHeight]);

  const wrapperStyle: CSSProperties = {
    minWidth: minWidth ?? 0,
    ...style,
  };
  const wrapperClassName = typeof className === "string" ? className : undefined;

  return (
    <div ref={hostRef} className={wrapperClassName} style={wrapperStyle}>
      {ready ? (
        <BaseResponsiveContainer
          {...props}
          width={width}
          height={height}
          minWidth={minWidth ?? 0}
          minHeight={minHeight}
          aspect={aspect}
        >
          {children}
        </BaseResponsiveContainer>
      ) : null}
    </div>
  );
}
