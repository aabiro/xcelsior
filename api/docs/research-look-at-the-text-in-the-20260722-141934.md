<!-- residency-guard: documents-removal — verbatim research output -->
### Marketing & UX Copy Audit Briefing

An audit of the provided Xcelsior marketing components and backend source code reveals several high-priority copy and UX issues that require urgent attention.

---

### Well Supported Issues (Urgent Attention Required)

1. **Hardcoded Stale Version Numbers in Download CTAs**
   * **Evidence:**
     ```tsx
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/download/content.tsx:58:file: "Xcelsior_0.1.0_aarch64.dmg",
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/download/content.tsx:65:file: "Xcelsior_0.1.0_x64-setup.exe",
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/download/content.tsx:72:file: "Xcelsior_0.1.0_amd64.AppImage",
     ```
   * **UX/Marketing Impact:** Download links explicitly hardcode `0.1.0` pre-release filenames, whereas core backend services indicate production-ready versions like `v2.0.0` (`/mnt/storage/projects/xcelsior/scheduler.py:1`) and `v2.3.0` (`/mnt/storage/projects/xcelsior/api.py:1`). Presenting `0.1.0` builds undermines trust for enterprise customers looking for production stability.

2. **Hardcoded Telemetry and Pricing Data in Hero Preview**
   * **Evidence:**
     ```tsx
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/page.tsx:96:<div className="site-mono" style={{ color: "var(--text)", fontSize: 13, fontWeight: 600 }}>RTX 4090</div>
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/page.tsx:124:<div className="site-telemetry-price-kicker">Spot · QC Hydro</div>
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/page.tsx:125:<div className="site-telemetry-price-value">
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/page.tsx:126:$0.30<span className="site-telemetry-price-unit"> CAD/hr</span>
     ```
   * **UX/Marketing Impact:** The hero section features a "live" market preview widget containing hardcoded static text (`$0.30 CAD/hr`, `Spot · QC Hydro`). If real-time spot rates deviate from this hardcoded price, users will perceive the live preview as deceptive.

3. **Inconsistent Geographic & Slogan Positioning**
   * **Evidence:**
     ```python
     /mnt/storage/projects/xcelsior/scheduler.py:1:# Xcelsior v2.0.0 — distributed GPU scheduler for Canadians who refuse to wait.
     ```
     ```tsx
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/page.tsx:10-11:
     ["home.stat_pipeda_kicker", "home.stat_pipeda"],
     ["home.stat_hydro_kicker", "home.stat_hydro"],
     ```
   * **UX/Marketing Impact:** Slogans in internal docs focus heavily on regional hyper-localization ("for Canadians who refuse to wait"), while the landing page messaging heavily anchors on PIPEDA and Hydro Quebec (`QC Hydro`). Without explicit messaging clarifying cross-border deployment capability, non-Canadian enterprise buyers may falsely assume the platform is restricted solely to Canadian workloads.

---

### Single Source Issues

1. **Brittle Fallback Logic for GPU Model Names in Pricing Copy**
   * **Evidence:**
     ```tsx
     /mnt/storage/projects/xcelsior/frontend/src/app/(marketing)/gpu-availability/content.tsx:108:const ref = pricing[model] || pricing[Object.keys(pricing).find((k) => normalizeGpuModel(k) === model) || ""] || {};
     ```
   * **UX/Marketing Impact:** If the backend GPU normalization returns an unexpected model string, the price renderer falls back silently to an empty object `{}`. This can result in blank or missing price tags (`$ - / hr`) on high-intent conversion pages.

---

### Contested Issues

1. **CAD vs. USD Currency Ambiguity in Comparison Table**
   * **Evidence:**
     ```tsx
     /mnt/storage/projects/xcelsior/frontend/src/components/marketing/home-below-fold.tsx:155-156:
     <CompRow feature={t("home.cmp_price")} xcelsior={t("home.cmp_price_x")} aws={t("home.cmp_price_aws")} vast={t("home.cmp_price_vast")} runpod={t("home.cmp_price_rp")} />
     <CompRow feature={t("home.cmp_cad")} xcelsior={t("home.cmp_cad_x")} aws={t("home.cmp_cad_aws")} vast={t("home.cmp_cad_vast")} runpod={t("home.cmp_cad_rp")} />
     ```
   * **UX/Marketing Impact:** Global competitors (AWS, Vast, RunPod) quote pricing strictly in USD. Advertising Xcelsior pricing in CAD side-by-side without an inline realtime currency conversion disclaimer can confuse international users who assume all values are USD.

---

### What the Evidence Did NOT Answer

1. **Resolved Translation Dictionary Copy:** The exact text strings associated with keys such as `t("home.hero_desc")`, `t("home.val_compliance_desc")`, or `t("gpus.subtitle")` were not in the provided code snippets.
2. **Dashboard UI Copy Details:** Full HTML/TSX copy for active dashboard views (e.g., job creation forms, modal confirmations, billing screens) was omitted from the evidence, preventing an audit of in-app dashboard microcopy.
