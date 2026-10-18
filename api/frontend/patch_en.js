const fs = require('fs');

function patchEnFile(filepath) {
  let content = fs.readFileSync(filepath, 'utf8');

  // Replace values
  content = content.replace(/"about\.val_green_title": "[^"]*",/g, '"about.val_tech_title": "Raw Performance",');
  content = content.replace(/"about\.val_green_desc": "[^"]*",/g, '"about.val_tech_desc": "Ultra-low-latency networking and hardware isolation ensure your models run at peak efficiency without noisy neighbors.",');

  content = content.replace(/"about\.val_community_title": "[^"]*",/g, '"about.val_community_title": "Community-Powered",');
  content = content.replace(/"about\.val_community_desc": "[^"]*",/g, '"about.val_community_desc": "Our marketplace connects GPU providers with AI teams across the globe. Providers earn, builders move faster, and the ecosystem grows.",');

  content = content.replace(/"about\.val_access_title": "[^"]*",/g, '"about.val_infra_title": "Developer Ease",');
  content = content.replace(/"about\.val_access_desc": "[^"]*",/g, '"about.val_infra_desc": "Ship faster with native MCP integration and a platform that gets out of your way. API-first, serverless-ready, and zero friction.",');

  content = content.replace(/"about\.val_canada_title": "[^"]*",/g, '"about.val_price_title": "Transparent Pricing",');
  content = content.replace(/"about\.val_canada_desc": "[^"]*",/g, '"about.val_price_desc": "No hyperscaler markup, no hidden egress fees. Straightforward spot and reserved compute pricing that scales with your needs.",');

  // Remove local
  content = content.replace(/[ \t]*"about\.val_local_title": "[^"]*",\n/g, '');
  content = content.replace(/[ \t]*"about\.val_local_desc": "[^"]*",\n/g, '');

  // Replace journey
  content = content.replace(/"about\.journey_2024_title": "[^"]*",/g, '"about.journey_2025_title": "2025",');
  content = content.replace(/"about\.journey_2024_p1": "[^"]*",/g, '"about.journey_2025_p1": "Xcelsior founded with a mission to build a transaction-safe GPU compute layer for the agentic era.",');
  content = content.replace(/"about\.journey_2024_p2": "[^"]*",/g, '"about.journey_2025_p2": "First GPU marketplace launch, bringing high-performance RTX 3090, 4090, A100, and H100 clusters online.",');

  content = content.replace(/"about\.journey_2025_title": "[^"]*",/g, '"about.journey_2025_p3": "Dynamic spot pricing and reserved compute plans launch globally.",');
  content = content.replace(/[ \t]*"about\.journey_2025_p1": "[^"]*",\n/g, '');
  content = content.replace(/[ \t]*"about\.journey_2025_p2": "[^"]*",\n/g, '');
  content = content.replace(/[ \t]*"about\.journey_2025_p3": "[^"]*",\n/g, '');

  content = content.replace(/"about\.journey_2026_title": "[^"]*",/g, '"about.journey_2026_title": "2026",');
  content = content.replace(/"about\.journey_2026_p1": "[^"]*",/g, '"about.journey_2026_p1": "MCP platform and agent-native tooling shipped, turning natural language into serverless compute.",');

  fs.writeFileSync(filepath, content);
}

patchEnFile('src/lib/i18n/en.ts');
patchEnFile('src/lib/i18n/en-public.ts');
