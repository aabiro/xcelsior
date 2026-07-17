"use client";

import { useEffect, useRef } from "react";

const SPRITES = [
  "/particles/particle-cyan.svg",
  "/particles/particle-purple.svg",
  "/particles/particle-emerald.svg",
] as const;

const WEIGHTS = [0.42, 0.42, 0.16] as const;

type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  depth: number;
  image: number;
  opacity: number;
  phase: number;
  pulseSpeed: number;
};

function randomBetween(min: number, max: number) {
  return min + Math.random() * (max - min);
}

function pickSprite() {
  const value = Math.random();
  let accumulated = 0;

  for (let index = 0; index < WEIGHTS.length; index += 1) {
    accumulated += WEIGHTS[index];
    if (value <= accumulated) return index;
  }

  return 0;
}

function makeParticles(count: number, width: number, height: number): Particle[] {
  return Array.from({ length: count }, () => {
    const depth = Math.random();
    const size = randomBetween(8, 34) * (0.5 + depth);

    return {
      x: Math.random() * width,
      y: Math.random() * height,
      vx: randomBetween(-0.14, 0.14) * (0.4 + depth),
      vy: randomBetween(-0.14, 0.14) * (0.4 + depth),
      size,
      depth,
      image: pickSprite(),
      opacity: randomBetween(0.25, 0.9) * (0.45 + depth * 0.6),
      phase: Math.random() * Math.PI * 2,
      pulseSpeed: randomBetween(0.0006, 0.0016),
    };
  });
}

type PixelFieldProps = {
  count?: number;
  className?: string;
  position?: "absolute" | "fixed";
};

/**
 * Canvas particle field based directly on the site-assets particle demo.
 * Particle depth drives size, velocity, opacity, and eased pointer parallax;
 * the SVG sprites are composited additively to retain their original glow.
 */
export function PixelField({ count, className, position = "absolute" }: PixelFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const images = SPRITES.map((source) => {
      const image = new Image();
      image.src = source;
      return image;
    });

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let dpr = 1;
    let pointerX = 0;
    let pointerY = 0;
    let targetPointerX = 0;
    let targetPointerY = 0;
    let particles: Particle[] = [];
    let animationFrame = 0;
    let disposed = false;

    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      width = bounds.width;
      height = bounds.height;
      dpr = Math.min(window.devicePixelRatio || 1, 2);

      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);

      if (particles.length === 0 && width > 0 && height > 0) {
        const particleCount = count ?? Math.round(Math.min(65, Math.max(32, (width * height) / 26000)));
        particles = makeParticles(particleCount, width, height);
      }
    };

    const draw = (now: number, advance: boolean) => {
      context.clearRect(0, 0, width, height);
      pointerX += (targetPointerX - pointerX) * 0.05;
      pointerY += (targetPointerY - pointerY) * 0.05;
      context.globalCompositeOperation = "lighter";

      for (const particle of particles) {
        if (advance) {
          particle.x += particle.vx;
          particle.y += particle.vy;
        }

        const margin = particle.size;
        if (particle.x < -margin) particle.x = width + margin;
        if (particle.x > width + margin) particle.x = -margin;
        if (particle.y < -margin) particle.y = height + margin;
        if (particle.y > height + margin) particle.y = -margin;

        const pulse = 0.72 + 0.28 * Math.sin(particle.phase + now * particle.pulseSpeed);
        const parallax = 26 * particle.depth;
        const x = particle.x + pointerX * parallax;
        const y = particle.y + pointerY * parallax;
        const image = images[particle.image];

        if (image.complete && image.naturalWidth) {
          context.globalAlpha = Math.min(1, particle.opacity * pulse);
          const size = particle.size * (0.9 + 0.1 * pulse);
          context.drawImage(image, x - size / 2, y - size / 2, size, size);
        }
      }

      context.globalAlpha = 1;
      context.globalCompositeOperation = "source-over";
    };

    const frame = (now: number) => {
      if (disposed) return;
      draw(now || 0, true);
      animationFrame = window.requestAnimationFrame(frame);
    };

    // Backgrounded tabs gain nothing from an animated wallpaper — stop spending
    // CPU/GPU on it while hidden, and pick back up cleanly on return.
    const handleVisibilityChange = () => {
      if (reduceMotion) return;
      if (document.hidden) {
        window.cancelAnimationFrame(animationFrame);
      } else if (!disposed) {
        animationFrame = window.requestAnimationFrame(frame);
      }
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (width <= 0 || height <= 0) return;
      const bounds = canvas.getBoundingClientRect();
      targetPointerX = ((event.clientX - bounds.left) / width - 0.5) * 2;
      targetPointerY = ((event.clientY - bounds.top) / height - 0.5) * 2;
    };

    const resetPointer = () => {
      targetPointerX = 0;
      targetPointerY = 0;
    };

    const drawStatic = () => {
      if (!disposed) draw(0, false);
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    window.addEventListener("pointerleave", resetPointer);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    if (reduceMotion) {
      images.forEach((image) => image.addEventListener("load", drawStatic));
      drawStatic();
    } else if (!document.hidden) {
      animationFrame = window.requestAnimationFrame(frame);
    }

    return () => {
      disposed = true;
      window.cancelAnimationFrame(animationFrame);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerleave", resetPointer);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      images.forEach((image) => image.removeEventListener("load", drawStatic));
    };
  }, [count]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className={`${position} inset-0 block h-full w-full pointer-events-none ${className ?? ""}`}
    />
  );
}
