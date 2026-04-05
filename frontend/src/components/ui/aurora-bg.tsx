"use client";

export function AuroraBackground({ className }: { className?: string }) {
  return (
    <div className={`pointer-events-none absolute inset-0 overflow-visible ${className ?? ""}`}>
      {/* Primary aurora blob - sits at top edge touching content */}
      <div
        className="absolute -top-4 left-1/4 h-[500px] w-[600px] rounded-full opacity-25"
        style={{
          background: "radial-gradient(ellipse, rgba(0,212,255,0.35) 0%, rgba(124,58,237,0.18) 50%, transparent 70%)",
          animation: "aurora-drift 8s ease-in-out infinite",
          filter: "blur(80px)",
        }}
      />
      {/* Secondary aurora blob */}
      <div
        className="absolute -top-8 right-1/4 h-[400px] w-[500px] rounded-full opacity-20"
        style={{
          background: "radial-gradient(ellipse, rgba(124,58,237,0.28) 0%, rgba(220,38,38,0.12) 50%, transparent 70%)",
          animation: "aurora-drift 12s ease-in-out infinite reverse",
          filter: "blur(100px)",
        }}
      />
      {/* Subtle green accent */}
      <div
        className="absolute top-0 left-1/2 h-[300px] w-[400px] rounded-full opacity-12"
        style={{
          background: "radial-gradient(ellipse, rgba(16,185,129,0.22) 0%, transparent 60%)",
          animation: "aurora-drift 10s ease-in-out infinite 2s",
          filter: "blur(60px)",
        }}
      />
    </div>
  );
}
