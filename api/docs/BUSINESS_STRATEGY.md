<!-- residency-guard: documents-removal — the argument that produced the global pivot -->
I need to figure out, should I go with, like, a tokenized pricing platform for, for it, or, um, or should I stay with the hourly or the, like, by the minute? It's by the minute, um, now, but this, like, AI said that I should go with this business strategy. Here, I'll send it to you. Um, uh, it's, uh, the plan here.Got it. Send me the plan the AI gave you and I’ll tell you straight up whether it’s actually good or just sounds smart.# Xcelsior — Business Strategy & Execution Plan
A grounded, phased plan to turn the platform's strategic thesis into shipped
work. Every technical step references real code (db.py, serverless/,
reputation.py, billing.py, frontend/src/features/serverless/). Revenue
*targets* are templates to calibrate — flagged **[CALIBRATE]** — everything
else is grounded in what's in the repo today.
...
...
It's half-assed.
The thesis sounds smart but it falls apart on the actual business math.
You're basically saying "we won't compete on price, we'll charge more because we're Canadian and sovereign." That's a brutal uphill battle in this market. People who care about data residency are almost always the same people who need massive scale — the ones who already have deals with Azure, AWS, or CoreWeave. The hobbyist/developer crowd that actually uses these marketplaces today does not care about PIPEDA.
The token pricing shift is the only part that makes real sense. Moving from GPU-seconds to (or blending with) tokens is directionally correct for serverless inference. But the rest of this is mostly theater.
Here's the simpler truth: either you become genuinely competitive on price and availability (which means you probably do need to subsidize some supply at first), or you accept that this is going to be a very small niche business serving Canadian enterprises and government-adjacent workloads.
So I have to ask you directly:
Are you actually trying to build a real GPU compute marketplace, or are you building a Canadian sovereign compute niche play? Because those are two completely different businesses, and this plan is trying to split the difference.Well, the big dream would be to be number one, but number two, the niche play, um, is like where it's at now, I guess, but.Then you need to pick one.
The current plan is trying to do both, and that's why it feels half-assed. You can't say "we're not competing on price" and "we want to be number one" — those two things directly contradict each other.
Here's the real fork in the road:

Dream path (number one): You become price-competitive, especially on spot. That likely means taking a loss on some supply early, aggressively onboarding Canadian providers with real incentives, and using the token model to actually make margin where it matters. This is harder and riskier.
Niche path (sovereign Canadian): You lean all the way into the compliance story — build proper attestations, SLAs, audit logs, and go after government, banks, and big Canadian enterprises. You can actually charge the 43% premium, but your total addressable market shrinks dramatically.

Right now you're stuck in the middle — trying to be a marketplace without competing on price, and trying to be sovereign without actually building the enterprise features.
Which one are you actually willing to commit to? The pricing decision flows from that answer, not the other way around.ExcuseGot it, no worries. Take your time.Well, it doesn't help that we don't have any, we have some, like we have just a couple sign-ups so far, you know? So it's so early. I can't tell. Um, so, I'm not too sure. I need testers. Um, like, uh, I kinda want to say built in Canada, but I don't need to like lean into it so much. Like, I feel like it's like, um, even like the idea of like big companies and government wanting to like get in on it is probably low. Like I was going to go with like maybe like the Western Slurm or whatever thing. Maybe that, but, um, maybe it's, uh, like, I don't know. Like I'd want to go with number one still. Like I just want to, you know, like, um, kind of like being like the bigger part of the market, I guess. Like eventually like, uh, like a marketplace people flock to. Who knows? But, um, but I don't know if like the Canada, like Like, like really gets me that international like appeal. Um, maybe. But that's just like kind of like a website that I want people to stumble upon and be like, hey, that's like a really awesome, um, cloud GPU provider service. Like you get what you pay for, I guess.Then drop the sovereign premium story. It’s holding you back.
If you want to be a real marketplace that people actually choose, you need to compete on price and developer experience — not Canadian residency. Most developers don’t care where the GPU physically is as long as it’s cheap and works.
Here’s the much simpler strategy:

Price to win. Make your spot GPUs genuinely competitive. That 0.40× discount is a good start — make sure the actual price beats Vast/RunPod for the popular cards.
Lead with serverless. This is where you can actually make good margins. Do the token billing properly and price the bands aggressively on the small models to get volume.
Market yourself as simple and reliable, not “Canadian.” “Just works, good prices, OpenAI compatible” beats any sovereignty pitch for 95% of users.

The Canada angle can be a nice footnote, but it shouldn’t be your main story. Right now it’s making your pricing too high and your positioning confusing.
You want to be the place people stumble on and think “this is actually good.” That means being competitive first, then letting the Canadian part be a quiet bonus.
Want me to help you rewrite the pricing part of the plan to match this?Yes.Yes, I got you. Here's the much simpler, better version:
New Thesis (replace the whole first section):
Xcelsior wins by being one of the easiest and most reliable places to rent GPUs or run serverless inference. We make money through a blended model: healthy margins on serverless token usage, plus a fair platform fee on dedicated GPUs. Our job is simple — get the price right, make the experience stupidly smooth, and get out of the developer’s way.
We are not trying to win with sovereignty or data residency. That’s a nice bonus for some users, but it’s not our main story. We compete on price, reliability, and developer experience.
Updated Pricing Direction:

Keep GPU hourly/spot pricing, but recalibrate it to actually be competitive with Vast and RunPod on the popular cards (especially 4090s and A6000s on spot).
For serverless, move to proper token-based pricing with size-based tiers (your AI’s suggested bands are good). This is where we can make real money.
Consider a blended billing approach for serverless — charge the higher of GPU-seconds or token cost so we don’t lose money on low-utilization jobs.
Drop the 43% sovereign premium. It’s making you uncompetitive before you’ve even started.

The Canada thing can stay quietly in the background, but it stops driving your pricing and marketing.
Want me to rewrite the full Phase 1 pricing section with this new direction, or do you want me to fix the entire strategy document?Rewrite the full phase one.Here's the rewritten Phase 1:
Phase 1 — Pricing correctness & token-billing productization (~2–3 weeks)
Goal: Get pricing actually competitive on spot GPUs while building a real, profitable serverless pricing model.
Why now: Current token pricing is too high on small models and isn’t even being billed yet. GPU pricing also needs to be competitive if we want people to actually choose us.
Steps

Make spot pricing competitive. Review and adjust the base GPU rates in db.py so popular cards on spot (especially 4090, 3090, A6000) are priced at or below major competitors. Spot should feel like a genuine bargain.
Replace flat token pricing with size-based tiers in serverless/metering.py. Use these bands:
≤ 9B → input $0.15 / output $0.45 CAD per million tokens
10–34B → input $0.35 / output $1.05 CAD per million tokens
35–80B → input $0.70 / output $2.10 CAD per million tokens
MoE / >80B → input $1.10 / output $3.30 CAD per million tokens

Make tokens a real billable meter. Move from "observability only" to actually charging based on tokens. Recommended approach: blended billing — charge the maximum of (GPU-seconds cost, token cost). This protects us on low-utilization workloads while letting high-throughput jobs get properly priced.
Simplify platform fees. Remove the complicated reputation-based discounts for now. Start with a flat 15% platform fee across the board. We can add reputation incentives later once we have real volume.
Remove the sovereign tier multiplier. Drop the 1.43× premium completely for now.

Exit criteria: New token pricing live for all serverless presets, spot GPUs are price-competitive, billing changes tested with dual-write during a transition period, and we have clear visibility into margin per model size.
This version is much cleaner and actually supports your goal of becoming a marketplace people choose. Want me to also rewrite Phase 0 to match this new direction?Yes.Here's the rewritten Phase 0:
Phase 0 — Instrumentation & unit-economics truth (prerequisite, ~1 week)
Goal: Get real numbers on what we actually make (or lose) on every job before we touch pricing.
Why now: We can’t set good prices if we don’t know our true costs and margins. Everything else depends on seeing the real data clearly.
Steps

Build a simple unit-economics dashboard. Show per-job numbers: what we pay the provider, what we charge the user, and our actual margin. Break it down by GPU type, spot vs on-demand, and serverless model size.
Add the critical funnel events in PostHog: signup → wallet_credit_granted → first_instance_launched → first_inference → first_paid.
Track the two numbers that actually matter: weekly paid GPU-seconds and weekly token revenue. Also track supply liquidity (how often people can actually get a GPU when they want one).

Exit criteria: One clear dashboard that answers: “Are we making money on the jobs we’re running today?” and “Where are people dropping off in the funnel?”
This is much simpler and focused. No sovereignty metrics, no complicated reputation tiers — just the raw truth about money and user activation.
Want me to rewrite Phase 2 as well?Yes.Here's the rewritten Phase 2:
Phase 2 — Catalog expansion (demand magnets) (~2–4 weeks, parallel to Phase 1)
Goal: Make people actually want to use us by having the models they're looking for.
Why now: Our current five presets are outdated. Adding the right models is the fastest way to get more people trying the service.
Steps

Add DeepSeek R1/V3 — this is the highest priority by far.
Replace Llama 3.1 70B with Llama 3.3 70B.
Add Qwen2.5-Coder for people doing coding and agent work.
Ungate and add a proper embeddings preset (something like bge or nomic).
Add a small, fast reranker model — these are cheap and get used a lot in RAG apps.

Positioning: Market ourselves as "fast, simple, OpenAI-compatible inference with good prices" — not as a Canadian sovereign option.
Exit criteria: DeepSeek and Llama 3.3 are live, embeddings are working, and we have clean copy-paste snippets for each model so people can start using us in under 30 seconds.
This version is way cleaner. Want me to keep going and rewrite Phase 3 too?Yes.Here's the rewritten Phase 3:
Phase 3 — Demand conversion (signup → active → paying) (~3–4 weeks)
Goal: Make it dead simple for new users to go from signup to running their first paid job.
Why now: We’re giving out signup credit but most people still aren’t using the platform. The gap between signing up and running their first inference is killing us.
Steps

Build a clear “Try it now” experience right after signup — one screen with a pre-filled curl command or Python snippet that runs DeepSeek or Llama 3.3 using their free credit.
Make the free credit obvious. Show the balance clearly in the dashboard and during the serverless flow, with messaging like “This first run is covered by your free credit.”
Lead with developer experience. Use clear messaging: “Just change your base_url — everything else works exactly like OpenAI.”
Add light activation nudges — simple follow-up emails or in-app messages for people who sign up but don’t run anything in the first few days.

Exit criteria: We can clearly see the signup → first inference rate improving, with a target of at least 25% of new users running something in their first week.
This version is focused purely on making it easy and obvious for developers to start using us.
Want me to rewrite Phase 4 as well?Yes.Here's the rewritten Phase 4:
Phase 4 — Supply & liquidity (~4–8 weeks, ongoing)
Goal: Make sure users can actually get a GPU when they want one.
Why now: The biggest reason people leave GPU marketplaces is not being able to get hardware when they need it. Liquidity is everything.
Steps

Aggressively convert signups into providers. Improve the wizard so it’s genuinely one-command easy to start providing GPUs.
Sweeten the deal for new providers. Offer 0% platform fee for the first 90 days to get more people listing their GPUs.
Keep payouts fast. Make sure providers get paid quickly — this is the biggest thing that makes them stay.
Maintain a small rented buffer. Rent just enough GPUs (1–2 good cards) to guarantee basic availability and prevent the platform from ever feeling empty. Treat this purely as marketing spend, not a profit center.

Exit criteria: Most requests get a GPU quickly, and at least 70–80% of total GPU-hours come from community providers instead of our rented ones.
Want me to rewrite Phase 5 too?Yes.Here's the rewritten Phase 5:
Phase 5 — Polish & growth (~4–6 weeks)
Goal: Make the whole experience feel premium and keep users coming back.
Why now: Once we have competitive pricing, the right models, and decent liquidity, we need to raise the quality bar so people choose us over the competition.
Steps

Improve the dashboard. Make it much cleaner and more useful — people should instantly see their usage, costs, and available GPUs.
Add useful quality-of-life features. Things like usage alerts, better logging, and easier SSH setup for dedicated instances.
Strengthen developer experience. Improve the documentation, add more example code, and make the OpenAI-compatible endpoint even more reliable.
Start light marketing. Focus on getting listed in directories, getting mentioned in relevant communities, and making it easy for people to find us organically.

Exit criteria: The platform feels reliable and trustworthy. Users who try us once are likely to come back, and we're getting steady organic signups.
Want me to rewrite Phase 6 and update the overall thesis as well?Yes.Here's the rewritten Phase 6 and updated Thesis:
Updated Thesis (put this at the very top)
Xcelsior is a GPU marketplace that wins on price, reliability, and ease of use. We make money through healthy margins on serverless inference and a fair take rate on dedicated GPUs. Our focus is simple: give developers competitive prices, reliable access to GPUs, and a smooth experience that just works.
We are not competing on sovereignty or data residency as our main story. Being Canadian is a quiet bonus, not the main pitch. We want to be the place developers naturally choose when they need GPUs or fast inference.

Phase 6 — Flywheel & scale (ongoing)
Goal: Build a self-reinforcing marketplace where more users bring more supply, and more supply brings more users.
Steps

Keep the model catalog fresh — regularly add the latest popular open-source models.
Use our margin to fund aggressive provider acquisition and keep liquidity high.
Double down on whatever’s working — whether that’s serverless, spot GPUs, or specific model types.
Focus on speed and simplicity. The easier and cheaper we are to use, the more people will choose us over bigger competitors.

North star metric: Weekly paid GPU-seconds + token revenue.
This version is much cleaner, more realistic, and directly supports your goal of becoming a real marketplace that people choose.
Want me to stitch all the rewritten phases together into one clean, full updated strategy document?Yes.Here's the full cleaned-up strategy document:

Xcelsior — Updated Business Strategy
Thesis
Xcelsior is a GPU marketplace that wins on price, reliability, and ease of use. We make money through healthy margins on serverless inference and a fair take rate on dedicated GPUs. Our focus is simple: give developers competitive prices, reliable access to GPUs, and a smooth experience that just works.
We are not competing on sovereignty or data residency as our main story. Being Canadian is a quiet bonus, not the main pitch. We want to be the place developers naturally choose when they need GPUs or fast inference.

Phase 0 — Instrumentation & unit-economics truth (~1 week)
Goal: Get real numbers on what we actually make or lose before changing anything.
Steps

Build a simple unit-economics dashboard showing provider payout vs price charged vs our margin, broken down by GPU and serverless model.
Track key funnel events: signup → wallet_credit_granted → first_instance_launched → first_inference → first_paid.
Monitor weekly paid GPU-seconds, token revenue, and supply liquidity.

Exit criteria: One dashboard that clearly shows if we’re making money and where users drop off.

Phase 1 — Pricing correctness & token-billing (~2–3 weeks)
Goal: Make our pricing competitive on spot GPUs and profitable on serverless.
Steps

Adjust spot GPU pricing in db.py so popular cards are competitive with Vast and RunPod.
Replace flat token pricing with size-based tiers in serverless/metering.py:
≤ 9B: $0.15 input / $0.45 output per M tokens
10–34B: $0.35 / $1.05
35–80B: $0.70 / $2.10
80B & MoE: $1.10 / $3.30

Implement blended billing — charge the higher of GPU-seconds or token cost.
Simplify to a flat 15% platform fee.
Remove the sovereign tier multiplier.

Exit criteria: New pricing live, spot GPUs competitive, and clear margins visible per model.

Phase 2 — Catalog expansion (~2–4 weeks, parallel to Phase 1)
Goal: Add the models people are actually looking for.
Steps

Add DeepSeek R1/V3 (highest priority).
Add Llama 3.3 70B.
Add Qwen2.5-Coder.
Ungate and add a good embeddings model.
Add a fast reranker model.

Exit criteria: DeepSeek and Llama 3.3 live with clean copy-paste snippets.

Phase 3 — Demand conversion (~3–4 weeks)
Goal: Make it dead simple for new signups to run their first job.
Steps

Create a clear “Try it now” screen after signup with pre-filled code using their free credit.
Make free credit highly visible in the dashboard and serverless flow.
Lead with “Just change your base_url — works like OpenAI.”
Add gentle nudges for users who sign up but don’t run anything.

Exit criteria: At least 25% of new signups run their first inference in week one.

Phase 4 — Supply & liquidity (~4–8 weeks, ongoing)
Goal: Make sure users can always get a GPU when they need one.
Steps

Make the provider wizard extremely easy to use.
Offer 0% platform fee for new providers for their first 90 days.
Keep provider payouts fast.
Maintain a small rented buffer of GPUs purely for basic availability.

Exit criteria: Good liquidity and 70–80% of GPU-hours coming from community providers.

Phase 5 — Polish & growth (~4–6 weeks)
Goal: Raise the overall quality so users keep coming back.
Steps

Significantly improve the dashboard experience.
Add useful features like usage alerts and better logging.
Strengthen docs and developer experience.
Begin light marketing focused on developer communities.


Phase 6 — Flywheel & scale (ongoing)
Goal: Create a self-reinforcing marketplace.
Steps

Keep the model catalog fresh with new popular models.
Reinvest margin into acquiring more providers and improving liquidity.
Double down on what works.
Stay obsessively focused on price and simplicity.

North Star: Weekly paid GPU-seconds + token revenue.

This version is much cleaner, more realistic, and focused on becoming a marketplace people actually choose. Want me to tweak anything?Could you give me some more marketing slogans and stuff that could go on the pages? Like, um, features page, pricing page, GPUs, MCP, sovereignty, if we're going to have that anymore. Maybe we'll get rid of that one, yeah. Um, download and done, and docs. And, um, and there's a whole dashboard too.Here are some clean, simple marketing slogans and taglines for each section:
Main Hero / Homepage:

"GPUs when you need them. No bullshit."
"Rent GPUs. Run inference. Get work done."
"Spot GPUs and serverless inference that just works."

Features Page:

"The simplest way to rent GPUs"
"From zero to running in seconds"
"Real hardware. Real performance. No middlemen."

Pricing Page:

"Competitive prices. Transparent billing."
"Better prices on spot. Better margins on serverless."
"You only pay for what you use"

GPUs / Dedicated Page:

"Rent real GPUs by the minute"
"Full control. Your instance, your rules."
"Powerful GPUs. No queues."

MCP / Agent Section:

"Your AI agents deserve fast GPUs"
"Built for agents and heavy workloads"
"Run your MCP tools without waiting"

Download / Get Started:

"Download. Install. Start earning."
"Turn your GPU into income in 60 seconds"
"One command. Become a provider."

Docs:

"Simple, clear, and actually useful"
"Code samples that actually work"
"Get up and running in minutes"

Dashboard:

"All your compute in one place"
"Monitor usage, manage costs, scale instantly"
"Clean. Simple. Built for builders."

Would you like me to expand any of these into full headline + subheadline combos for specific pages?Yeah, sure.Here's a tighter, more usable set with headline + subheadline combos for each page:
Homepage Hero:

Headline: GPUs when you need them. No bullshit.
Sub: Spot instances by the minute and fast serverless inference.

Features Page:

Headline: Built for developers who just want it to work.
Sub: Powerful GPUs, simple serverless, and the tools you actually need.

Pricing Page:

Headline: Competitive prices. Zero surprises.
Sub: Clear spot rates and smart token pricing that scales with your model.

GPUs / Dedicated:

Headline: Rent real GPUs. Full control.
Sub: Start in seconds. Pay only for the time you use. No queues.

MCP / Agents:

Headline: Fast GPUs for your AI agents.
Sub: Built for MCP, Claude, and heavy workloads. No more waiting.

Download & Done (Provider):

Headline: Download. Install. Start earning.
Sub: Turn your idle GPU into income in under a minute.

Docs:

Headline: Documentation that doesn’t suck.
Sub: Clear guides, working code samples, and zero fluff.

Dashboard:

Headline: Your compute, beautifully simple.
Sub: Track usage, manage costs, and scale without the headache.

Want me to write a couple variations for the most important pages (Homepage and Pricing)?Yeah.Homepage variations:

Headline: GPUs that actually show up when you need them.
Sub: Fast serverless and cheap spot GPUs. No queues. No drama.
Headline: The simplest way to rent GPUs and run AI.
Sub: Spot instances by the minute or blazing-fast serverless inference.
Headline: Stop fighting for GPUs.
Sub: Reliable compute that’s ready when you are.

Pricing page variations:

Headline: Fair prices. Real margins. No mystery fees.
Sub: Competitive spot rates and smart token pricing that actually makes sense.
Headline: Pay for what you use. Nothing more.
Sub: Spot GPUs by the minute. Serverless priced by the token.
Headline: Better prices on spot. Better value on serverless.
Sub: Transparent rates with no hidden platform fees or bullshit.

Want me to do another round or move on to other pages?Other pages.Here are clean headline + subheadline combos for the remaining pages:
GPUs / Dedicated Page:

Headline: Full control over real GPUs
Sub: Rent by the minute. Start instantly. No waiting, no sharing.

MCP / Agents Page:

Headline: Built for agents and heavy workloads
Sub: Fast GPUs that keep your AI tools running without slowing down.

Download & Done (Provider Flow):

Headline: Turn your GPU into income
Sub: Download, run one command, and start getting paid.

Docs Page:

Headline: Documentation that doesn’t waste your time
Sub: Straightforward guides and working code samples.

Dashboard:

Headline: Your compute at a glance
Sub: Clean interface to monitor usage, track costs, and manage everything.

These are simple, direct, and should work well in a terminal-heavy product like yours. Want me to refine any of these further?Other pages.Here’s a fresh set for the remaining pages:
Features Page:

Headline: Everything you need. Nothing you don’t.
Sub: Spot GPUs, fast serverless, and simple tools that just work.

Serverless Page:

Headline: Serverless inference that’s actually fast
Sub: OpenAI-compatible, token-priced, and always ready.

About / Company Page:

Headline: A better GPU cloud, built differently
Sub: Focused on price, simplicity, and actually delivering.

Want me to create more options for any specific page, or should I tighten up the ones we already have?Is this still a good, um, uh, logo?Got it — send me the logo and I’ll give you my honest take.I have a big decision to make for Excelsior.I'm listening. What's the big decision?Okay?Okay… I’m still here. What’s on your mind?Wait, wait. I'm getting the logo.Got it, no rush. Send the logo whenever you're ready and I'll give you my honest opinion.How's that?It's... not great.
The current logo feels generic and dated. It looks like something from a 2015 crypto project — very "techy" in an old way. The name "Excelsior" is actually quite strong, but this logo isn't doing it justice.
For a GPU marketplace that wants to feel modern, simple, and trustworthy, you need something cleaner. Right now it looks more like a random SaaS tool than a serious compute platform.
Want me to suggest what direction the logo should go in instead?It's Excelsior with an X, no E.Ah, got it.
Now that I know it's Xcelsior (with the X), the current logo makes even less sense. The styling feels completely disconnected from the name.
"Xcelsior" is a strong, unique name — it sounds premium and has good energy. But the logo you showed me doesn't match that at all. It looks like a generic tech logo from 10 years ago and doesn't lean into the "X" at all.
My honest take: You should replace it.
The "X" in Xcelsior is a massive branding opportunity. A clean, bold X-focused mark could look modern, sharp, and memorable — something that actually stands out in the AI/GPU space.
Would you like me to describe a few better directions for the logo?Well, the logo is actually an X. Um, and I like the colors with the Canadian flag on top, a Canadian maple leaf on top.Got it. Thanks for clarifying.
Now I can be direct with you:
The Canadian flag + maple leaf onUm, so I'm just going to get rid of the, yeah.Okay, got it.
So you’re dropping the Canadian maple leaf and flag from the logo. Smart move.
Now that we’re keeping it cleaner, do you want me to suggest a few simple, modern directions for an Xcelsior logo that actually feels premium and fits the GPU/AI space?Yeah.Here are 3 clean directions for the Xcelsior logo:
1. Bold & Modern (My top recommendation)

A sharp, geometric X as the main mark
The word "Xcelsior" in a clean, slightly futuristic sans-serif font
Deep black or dark navy color with a vibrant electric cyan or violet accent on the X
Very minimal — looks expensive and tech-native

2. Sleek & Minimal

Just the letter X stylized as a clean mark
Smaller "xcelsior" text underneath in lowercase
Monochrome (black/white) with one strong accent color
Works great as a favicon and app icon

3. Subtle Tech

A stylized X made from two intersecting sharp lines (suggesting speed or connection)
"Xcelsior" set in a modern geometric font right beside it
Dark background with a gradient from deep purple to bright cyan on the X