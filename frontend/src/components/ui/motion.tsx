"use client";

import { motion, type Variants } from "framer-motion";
import type { HTMLAttributes, ReactNode } from "react";

// ── Fade-in wrapper ─────────────────────────────────────────────────

export function FadeIn({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ── Staggered list container + child ────────────────────────────────

const staggerContainer: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};

const staggerItem: Variants = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" } },
};

export function StaggerList({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      variants={staggerContainer}
      initial="hidden"
      animate="show"
      className={className}
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <motion.div variants={staggerItem} className={className}>
      {children}
    </motion.div>
  );
}

// ── Count-up stat ───────────────────────────────────────────────────

export function CountUp({
  value,
  prefix = "",
  suffix = "",
  duration = 1.2,
}: {
  value: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
}) {
  return (
    <motion.span
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <motion.span
        // Uses Framer's animate-values trick — animate a CSS variable and read it
        initial={{ "--count": 0 } as Record<string, number>}
        animate={{ "--count": value } as Record<string, number>}
        transition={{ duration, ease: "easeOut" }}
        style={{ counterSet: `count var(--count)` } as React.CSSProperties}
      >
        {/* Fall back to final value immediately — no counter-set browser support needed */}
      </motion.span>
      {prefix}{typeof value === "number" ? value.toLocaleString() : value}{suffix}
    </motion.span>
  );
}

// ── Hover card wrapper ──────────────────────────────────────────────

export function HoverCard({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      whileHover={{ y: -3, boxShadow: "0 8px 30px rgba(0,0,0,0.25)" }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ── Scroll-triggered reveal ─────────────────────────────────────────

export function ScrollReveal({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
