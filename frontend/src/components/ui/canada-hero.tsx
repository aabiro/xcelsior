"use client";

/**
 * High-fidelity inline SVG Canada map hero with provincial outlines,
 * glowing data-centre dots, inter-city connection arcs, and aurora effects.
 * Loads instantly — no external assets.
 */
export function CanadaMapHero({
  hostCount,
  className,
}: {
  hostCount: number;
  className?: string;
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-border bg-surface ${className ?? ""}`}
      style={{ minHeight: 220 }}
    >
      {/* Aurora glow layers */}
      <div className="pointer-events-none absolute inset-0" style={{ filter: "blur(70px)" }}>
        <div className="absolute left-[20%] -top-8 h-48 w-72 rounded-full bg-accent-cyan/12" />
        <div className="absolute left-[50%] top-2 h-36 w-56 rounded-full bg-accent-violet/10" />
        <div className="absolute right-[10%] top-12 h-28 w-44 rounded-full bg-accent-red/6" />
      </div>

      {/* ── Canada Map SVG ───────────────────────────────────────── */}
      <svg
        viewBox="0 0 1000 520"
        className="pointer-events-none absolute inset-0 h-full w-full"
        aria-hidden
      >
        <defs>
          {/* Aurora gradient for province fills */}
          <linearGradient id="canada-fill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#00d4ff" stopOpacity="0.06" />
            <stop offset="50%" stopColor="#7c3aed" stopOpacity="0.04" />
            <stop offset="100%" stopColor="#dc2626" stopOpacity="0.03" />
          </linearGradient>
          {/* Glow filter for city dots */}
          <filter id="city-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="3" />
          </filter>
          {/* Connection line gradient */}
          <linearGradient id="conn-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#00d4ff" stopOpacity="0.4" />
            <stop offset="50%" stopColor="#7c3aed" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#00d4ff" stopOpacity="0.4" />
          </linearGradient>
        </defs>

        {/* ── Provincial shapes ──────────────────────────────────── */}
        <g fill="url(#canada-fill)" stroke="currentColor" strokeWidth="0.8" opacity="0.15">
          {/* British Columbia */}
          <path d="M95 460 L90 430 L85 400 L80 370 L78 340 L82 310 L90 280 L100 255 L112 235 L120 220 L125 200 L128 180 L130 165 L135 150 L140 140 L148 125 L155 115 L160 100 L165 85 L170 75 L175 65 L185 60 L195 50 L200 42 L210 38 L218 35 L225 38 L230 45 L225 55 L220 65 L225 80 L235 75 L245 70 L255 65 L260 72 L255 85 L250 100 L255 115 L260 130 L258 150 L252 170 L248 190 L245 210 L243 235 L250 255 L258 275 L260 295 L255 320 L250 345 L248 370 L250 395 L252 420 L255 445 L258 460 Z" />
          {/* Alberta */}
          <path d="M258 460 L255 445 L252 420 L250 395 L248 370 L250 345 L255 320 L260 295 L258 275 L260 255 L262 235 L268 215 L275 195 L280 175 L282 155 L278 135 L275 120 L278 100 L285 85 L292 75 L300 68 L310 62 L318 58 L325 55 L330 60 L328 72 L325 88 L330 105 L335 120 L332 140 L328 160 L325 180 L328 200 L332 220 L338 240 L345 260 L348 280 L345 300 L342 320 L340 345 L342 370 L345 395 L348 420 L350 445 L352 460 Z" />
          {/* Saskatchewan */}
          <path d="M352 460 L350 445 L348 420 L345 395 L342 370 L340 345 L342 320 L345 300 L348 280 L352 260 L358 240 L365 220 L370 200 L375 180 L378 160 L375 140 L372 118 L375 98 L382 80 L390 65 L398 55 L408 48 L418 42 L428 38 L435 40 L432 52 L428 68 L432 85 L438 100 L442 118 L445 140 L442 160 L438 180 L435 200 L438 220 L442 240 L448 260 L450 280 L448 300 L445 320 L442 345 L445 370 L448 395 L450 420 L452 445 L455 460 Z" />
          {/* Manitoba */}
          <path d="M455 460 L452 445 L450 420 L448 395 L445 370 L442 345 L445 320 L448 300 L450 280 L455 260 L460 240 L465 220 L470 200 L472 180 L468 160 L465 140 L468 118 L475 95 L482 78 L490 62 L500 50 L510 42 L520 35 L530 30 L540 28 L548 32 L542 45 L535 62 L540 80 L548 95 L555 112 L558 130 L555 150 L550 170 L545 195 L548 215 L552 235 L558 255 L562 275 L560 300 L555 325 L552 350 L555 375 L558 400 L560 430 L562 460 Z" />
          {/* Ontario */}
          <path d="M562 460 L560 430 L558 400 L555 375 L552 350 L555 325 L560 300 L562 275 L568 255 L575 235 L580 215 L582 195 L578 175 L575 155 L580 135 L588 115 L595 98 L605 80 L615 65 L625 55 L638 48 L650 42 L665 38 L678 35 L690 32 L700 35 L705 42 L700 55 L692 70 L698 85 L708 95 L718 108 L720 125 L715 145 L708 165 L700 185 L698 208 L705 228 L712 248 L715 268 L710 290 L702 310 L695 330 L690 350 L688 368 L692 388 L698 405 L705 425 L710 440 L712 460 Z" />
          {/* Quebec */}
          <path d="M712 460 L710 440 L705 425 L698 405 L692 388 L688 368 L690 350 L695 330 L702 310 L710 290 L715 268 L718 248 L722 228 L728 208 L735 188 L740 168 L738 148 L735 128 L740 108 L748 88 L758 72 L768 58 L780 48 L792 40 L805 35 L818 32 L830 30 L840 32 L848 38 L845 50 L838 65 L842 80 L850 92 L858 105 L862 122 L858 142 L852 162 L848 185 L852 205 L858 225 L862 248 L858 272 L852 295 L845 318 L840 340 L842 365 L848 388 L852 410 L855 432 L858 460 Z" />
          {/* Atlantic Provinces (NB, NS, PEI, NL grouped) */}
          <path d="M858 460 L855 432 L852 410 L848 388 L845 368 L848 345 L855 325 L862 305 L868 288 L875 272 L880 258 L888 242 L895 228 L902 215 L908 200 L912 188 L918 178 L925 172 L932 168 L938 170 L942 178 L940 190 L935 205 L940 220 L948 235 L952 252 L948 270 L942 290 L935 310 L932 330 L935 350 L940 370 L945 390 L948 410 L950 435 L952 460 Z" />
          {/* Northern Territories (YT, NT, NU) — top band */}
          <path d="M95 460 L100 255 L135 150 L175 65 L225 38 L300 68 L398 55 L500 50 L540 28 L665 38 L780 48 L840 32 L918 178 L950 460"
                fill="none" strokeDasharray="4 6" opacity="0.08" />
        </g>

        {/* ── Provincial border highlights ───────────────────────── */}
        <g stroke="#00d4ff" strokeWidth="0.5" opacity="0.08">
          <line x1="258" y1="72" x2="258" y2="460" />   {/* BC/AB */}
          <line x1="352" y1="55" x2="352" y2="460" />   {/* AB/SK */}
          <line x1="455" y1="38" x2="455" y2="460" />   {/* SK/MB */}
          <line x1="562" y1="28" x2="562" y2="460" />   {/* MB/ON */}
          <line x1="712" y1="32" x2="712" y2="460" />   {/* ON/QC */}
          <line x1="858" y1="32" x2="858" y2="460" />   {/* QC/Atlantic */}
        </g>

        {/* ── Connection arcs (data-centre network) ──────────────── */}
        <g fill="none" stroke="url(#conn-grad)" strokeWidth="1" opacity="0.5">
          {/* Vancouver → Calgary */}
          <path d="M168 385 Q 240 340 325 370" />
          {/* Calgary → Toronto */}
          <path d="M325 370 Q 480 280 660 410" />
          {/* Toronto → Montreal */}
          <path d="M660 410 Q 700 370 745 385" />
          {/* Montreal → Halifax */}
          <path d="M745 385 Q 820 350 895 400" />
          {/* Vancouver → Toronto (long-haul) */}
          <path d="M168 385 Q 420 250 660 410" strokeDasharray="6 4" opacity="0.3" />
        </g>

        {/* ── City dots with glow ────────────────────────────────── */}
        <g>
          {/* Vancouver */}
          <circle cx="168" cy="385" r="6" fill="#7c3aed" opacity="0.15" filter="url(#city-glow)" />
          <circle cx="168" cy="385" r="3.5" fill="#7c3aed" opacity="0.9" />
          <circle cx="168" cy="385" r="8" fill="#7c3aed" opacity="0.12">
            <animate attributeName="r" values="8;14;8" dur="3.5s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.12;0.03;0.12" dur="3.5s" repeatCount="indefinite" />
          </circle>
          <text x="168" y="405" textAnchor="middle" fill="#7c3aed" fontSize="10" fontFamily="var(--font-sans)" opacity="0.6">Vancouver</text>

          {/* Calgary */}
          <circle cx="325" cy="370" r="5" fill="#00d4ff" opacity="0.12" filter="url(#city-glow)" />
          <circle cx="325" cy="370" r="3" fill="#00d4ff" opacity="0.85" />
          <circle cx="325" cy="370" r="7" fill="#00d4ff" opacity="0.1">
            <animate attributeName="r" values="7;12;7" dur="4s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.1;0.03;0.1" dur="4s" repeatCount="indefinite" />
          </circle>
          <text x="325" y="390" textAnchor="middle" fill="#00d4ff" fontSize="10" fontFamily="var(--font-sans)" opacity="0.6">Calgary</text>

          {/* Toronto — primary hub (larger) */}
          <circle cx="660" cy="410" r="8" fill="#00d4ff" opacity="0.18" filter="url(#city-glow)" />
          <circle cx="660" cy="410" r="4.5" fill="#00d4ff" opacity="0.95" />
          <circle cx="660" cy="410" r="12" fill="#00d4ff" opacity="0.15">
            <animate attributeName="r" values="12;20;12" dur="3s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.15;0.04;0.15" dur="3s" repeatCount="indefinite" />
          </circle>
          <text x="660" y="432" textAnchor="middle" fill="#00d4ff" fontSize="11" fontWeight="600" fontFamily="var(--font-sans)" opacity="0.7">Toronto</text>

          {/* Montreal */}
          <circle cx="745" cy="385" r="6" fill="#00d4ff" opacity="0.15" filter="url(#city-glow)" />
          <circle cx="745" cy="385" r="3.5" fill="#00d4ff" opacity="0.9" />
          <circle cx="745" cy="385" r="9" fill="#00d4ff" opacity="0.12">
            <animate attributeName="r" values="9;15;9" dur="3.8s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.12;0.03;0.12" dur="3.8s" repeatCount="indefinite" />
          </circle>
          <text x="745" y="405" textAnchor="middle" fill="#00d4ff" fontSize="10" fontFamily="var(--font-sans)" opacity="0.6">Montréal</text>

          {/* Halifax */}
          <circle cx="895" cy="400" r="4" fill="#10b981" opacity="0.1" filter="url(#city-glow)" />
          <circle cx="895" cy="400" r="2.5" fill="#10b981" opacity="0.8" />
          <circle cx="895" cy="400" r="6" fill="#10b981" opacity="0.1">
            <animate attributeName="r" values="6;10;6" dur="4.5s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.1;0.03;0.1" dur="4.5s" repeatCount="indefinite" />
          </circle>
          <text x="895" y="418" textAnchor="middle" fill="#10b981" fontSize="9" fontFamily="var(--font-sans)" opacity="0.5">Halifax</text>

          {/* Edmonton */}
          <circle cx="308" cy="330" r="2" fill="#00d4ff" opacity="0.7" />
          <circle cx="308" cy="330" r="5" fill="#00d4ff" opacity="0.08">
            <animate attributeName="r" values="5;8;5" dur="5s" repeatCount="indefinite" />
          </circle>

          {/* Winnipeg */}
          <circle cx="485" cy="395" r="2" fill="#00d4ff" opacity="0.7" />
          <circle cx="485" cy="395" r="5" fill="#00d4ff" opacity="0.08">
            <animate attributeName="r" values="5;8;5" dur="4.2s" repeatCount="indefinite" />
          </circle>

          {/* Ottawa */}
          <circle cx="700" cy="400" r="2" fill="#00d4ff" opacity="0.7" />
          <circle cx="700" cy="400" r="5" fill="#00d4ff" opacity="0.08">
            <animate attributeName="r" values="5;8;5" dur="3.7s" repeatCount="indefinite" />
          </circle>
        </g>
      </svg>

      {/* ── Text overlay ─────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-col items-start justify-center p-6 sm:p-8 h-full" style={{ minHeight: 220 }}>
        <div className="flex items-center gap-2 mb-3">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald animate-pulse" />
          <span className="text-xs font-medium uppercase tracking-wider text-emerald">Online</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
          <span className="text-accent-cyan">{hostCount}</span>{" "}
          <span className="text-text-primary">Canadian Host{hostCount === 1 ? "" : "s"}</span>
        </h2>
        <p className="mt-2 text-sm text-text-secondary max-w-md">
          GPU compute nodes distributed across Canada&apos;s provinces — connected, sovereign, and ready&nbsp;for&nbsp;AI&nbsp;workloads
        </p>
        <div className="mt-4 flex flex-wrap gap-3 text-[11px] text-text-muted">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent-cyan" />
            Primary Hub
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent-violet" />
            West Coast
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald" />
            Atlantic
          </span>
        </div>
      </div>
    </div>
  );
}
