import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Xcelsior Pricing — Save up to 60% on GPU Compute";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function PricingOG() {
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

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            marginBottom: 24,
          }}
        >
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: 12,
              background: "#dc2626",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 32,
              fontWeight: 800,
              color: "white",
            }}
          >
            X
          </div>
          <span style={{ fontSize: 36, fontWeight: 700, color: "#f8fafc" }}>
            Xcelsior
          </span>
        </div>

        <p
          style={{
            fontSize: 48,
            fontWeight: 800,
            color: "#f8fafc",
            marginBottom: 16,
            marginTop: 0,
          }}
        >
          GPU Pricing
        </p>
        <p
          style={{
            fontSize: 24,
            color: "#94a3b8",
            marginTop: 0,
            marginBottom: 40,
          }}
        >
          Up to 60% less than hyperscalers • Pay in CAD
        </p>

        {/* GPU price cards */}
        <div style={{ display: "flex", gap: 24 }}>
          {[
            { gpu: "RTX 4090", price: "$0.45/hr" },
            { gpu: "A100 80GB", price: "$1.89/hr" },
            { gpu: "H100 SXM", price: "$2.99/hr" },
          ].map((card) => (
            <div
              key={card.gpu}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                padding: "20px 32px",
                borderRadius: 12,
                border: "1px solid #334155",
                background: "#1e293b",
              }}
            >
              <span style={{ fontSize: 18, color: "#f59e0b", fontWeight: 600 }}>
                {card.gpu}
              </span>
              <span
                style={{ fontSize: 28, fontWeight: 800, color: "#f8fafc", marginTop: 8 }}
              >
                {card.price}
              </span>
            </div>
          ))}
        </div>

        <p
          style={{ position: "absolute", bottom: 32, fontSize: 20, color: "#64748b" }}
        >
          xcelsior.ca/pricing
        </p>
      </div>
    ),
    { ...size }
  );
}
