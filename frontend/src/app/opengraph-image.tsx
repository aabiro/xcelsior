import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Xcelsior — Canada-First GPU Compute for Teams Worldwide";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          background: "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)",
          fontFamily: "sans-serif",
        }}
      >
        {/* Top accent bar */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 6,
            background: "linear-gradient(90deg, #dc2626, #f59e0b, #dc2626)",
          }}
        />

        {/* Logo badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 20,
            marginBottom: 32,
          }}
        >
          <div
            style={{
              width: 80,
              height: 80,
              borderRadius: 16,
              background: "#dc2626",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 48,
              fontWeight: 800,
              color: "white",
            }}
          >
            X
          </div>
          <span
            style={{
              fontSize: 56,
              fontWeight: 700,
              color: "#f8fafc",
              letterSpacing: -1,
            }}
          >
            Xcelsior
          </span>
        </div>

        {/* Tagline */}
        <p
          style={{
            fontSize: 28,
            color: "#94a3b8",
            marginTop: 0,
            marginBottom: 40,
          }}
        >
          Canada-First GPU Compute for Teams Worldwide
        </p>

        {/* Feature pills */}
        <div style={{ display: "flex", gap: 16 }}>
          {["Transparent Pricing", "Compliance-Aware", "Teams Worldwide"].map(
            (label) => (
              <div
                key={label}
                style={{
                  padding: "10px 24px",
                  borderRadius: 999,
                  border: "1px solid #334155",
                  color: "#f59e0b",
                  fontSize: 18,
                  fontWeight: 600,
                }}
              >
                {label}
              </div>
            )
          )}
        </div>

        {/* Domain */}
        <p
          style={{
            position: "absolute",
            bottom: 32,
            fontSize: 20,
            color: "#64748b",
          }}
        >
          xcelsior.ca
        </p>
      </div>
    ),
    { ...size }
  );
}
