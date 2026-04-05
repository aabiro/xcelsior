"use client";

export function AuroraBackground({ className }: { className?: string }) {
  return (
    <div className={`pointer-events-none absolute -top-6 left-0 right-0 bottom-0 ${className ?? ""}`}>
      {/* Primary aurora blob */}
      <div
        className="absolute -top-16 left-1/4 h-[500px] w-[600px] rounded-full opacity-20"
        style={{
          background: "radial-gradient(ellipse, rgba(0,212,255,0.3) 0%, rgba(124,58,237,0.15) 50%, transparent 70%)",
          animation: "aurora-drift 8s ease-in-out infinite",
          filter: "blur(80px)",
        }}
      />
      {/* Secondary aurora blob */}
      <div
        className="absolute -top-20 right-1/4 h-[400px] w-[500px] rounded-full opacity-15"
        style={{
          background: "radial-gradient(ellipse, rgba(124,58,237,0.25) 0%, rgba(220,38,38,0.1) 50%, transparent 70%)",
          animation: "aurora-drift 12s ease-in-out infinite reverse",
          filter: "blur(100px)",
        }}
      />
      {/* Subtle green accent */}
      <div
        className="absolute -top-8 left-1/2 h-[300px] w-[400px] rounded-full opacity-10"
        style={{
          background: "radial-gradient(ellipse, rgba(16,185,129,0.2) 0%, transparent 60%)",
          animation: "aurora-drift 10s ease-in-out infinite 2s",
          filter: "blur(60px)",
        }}
      />
    </div>
  );
}
