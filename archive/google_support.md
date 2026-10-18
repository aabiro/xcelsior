• Confirmed: digital-circuit-hkw4x is inside source organization 71994629047, four folders deep. You cannot complete this migration yourself because you lack permission on source
  folder 21358131809.

  The required migration handshake is:

  - Source org 71994629047: allow export to under:organizations/812964255343.
  - Destination org 812964255343: allow import from under:organizations/71994629047.
  - Source folder 21358131809: grant Project Mover to your identity.
  - Destination org: grant Project Creator.
  - Project: Project IAM Admin or Owner.

  Project ownership alone cannot override the source-folder requirement. Google migration requirements (https://docs.cloud.google.com/resource-manager/docs/perform-migration)

  Send Google/Startup Support this exact request:

  > Project digital-circuit-hkw4x (480880453449) currently resides under folder 21358131809 in organization 71994629047.
  > Please authorize its export to organization 812964255343 by allowing under:organizations/812964255343 in constraints/resourcemanager.allowedExportDestinations, and grant
  > aaryn.biro@xcelsior.ca Project Mover on the source folder or perform the move for us.
  > Preserve billing account 0149F8-DB6353-622382 and Google AI Studio Tier 3. Do not alter billing account 01E88E-11F37C-ADCBD5, which holds our startup credits.

  On your destination, keep the import permission narrowly limited to:

  under:organizations/71994629047

  Do not allow all sources, and remove that exception after migration.

  Until Google authorizes export, keep using API credentials created in digital-circuit-hkw4x; your workloads may run inside xcelsior.ca while Gemini usage remains billed through the
  Tier 3 project. Your $2,000 account remains untouched.

  Hello Google Cloud / Startup Support,
  >
  > Please approve the pending IAM access request submitted on July 25, 2026 for:
  >
  > - User: aaryn.biro@xcelsior.ca
  > - Project: digital-circuit-hkw4x
  > - Project number: 480880453449
  > - Current source organization: 71994629047
  > - Current parent folder: 21358131809
  > - Billing account: 0149F8-DB6353-622382
  > - Google AI Studio status: Imported, Tier 3
  >
  > I need to create a current Google AI Studio authorization API key in this project. Please grant the following least-privilege roles directly on digital-circuit-hkw4x:
  >
  > - Service Usage Admin: roles/serviceusage.serviceUsageAdmin
  > - API Keys Admin: roles/serviceusage.apiKeysAdmin
  > - Service Account Creator: roles/iam.serviceAccountCreator
  > - Service Account API Key Binding Admin: roles/iam.serviceAccountApiKeyBindingAdmin
  > - Security Reviewer: roles/iam.securityReviewer
  >
  > These grants are required for:
  >
  > - serviceusage.services.enable
  > - apikeys.keys.create
  > - iam.serviceAccounts.create
  > - iam.serviceAccountApiKeyBindings.create
  > - resourcemanager.projects.get
  > - resourcemanager.projects.getIamPolicy
  >
  > Please also enable generativelanguage.googleapis.com on the project if Google policy prevents granting Service Usage Admin.
  >
  > Please verify that no IAM deny policy, Principal Access Boundary, organization policy, or access policy blocks these permissions.
  >
  > Important safeguards:
  >
  > - Preserve billing association with 0149F8-DB6353-622382.
  > - Preserve the project’s Google AI Studio Tier 3 status.
  > - Do not change or close the Google-managed billing account.
  > - Do not modify billing account 01E88E-11F37C-ADCBD5, which holds our startup promotional credits.
  > - Do not change the project’s parent hierarchy as part of this IAM request.
  >
  > If any requested predefined role cannot be granted on this Google-managed project, please grant an equivalent custom role containing the listed permissions or create the
  > authorization key through a secure Google-supported handoff.
  >
  > Please confirm each granted role and any remaining deny/PAB restrictions.
  >
  > Thank you,
  >
  > Aaryn Biro
  > aaryn.biro@xcelsior.ca

    You cannot enable the API, create/list/retrieve keys, create the authorization-key service account, bind it, or modify IAM. There is no self-service grant path because you lack
  both getIamPolicy and setIamPolicy.

  Send this updated request:

  Subject: Temporary AI Studio authorization-key permissions for Tier 3 project

  > Please temporarily grant aaryn.biro@xcelsior.ca these roles directly on project digital-circuit-hkw4x (480880453449):
  >
  > - roles/serviceusage.serviceUsageAdmin
  > - roles/serviceusage.apiKeysAdmin
  > - roles/iam.serviceAccountCreator
  > - roles/iam.serviceAccountApiKeyBindingAdmin
  >
  > A projects.testIamPermissions audit confirms I currently possess only:
  >
  > - resourcemanager.projects.get
  > - serviceusage.services.get
  > - serviceusage.services.list
  >
  > I am missing:
  >
  > - serviceusage.services.enable
  > - apikeys.keys.create
  > - apikeys.keys.list
  > - apikeys.keys.getKeyString
  > - apikeys.keys.update
  > - iam.serviceAccounts.create
  > - iam.serviceAccounts.list
  > - iam.serviceAccountApiKeyBindings.create
  >
  > These permissions are required by the current Google AI Studio authorization-key workflow.
  >
  > Please also ensure generativelanguage.googleapis.com is enabled. The roles can be revoked after I create, securely store, and verify the authorization key.
  >
  > Alternatively, please create an authorization key on my behalf and provide it through a secure secret-delivery mechanism—not ordinary email.
  >
  > Preserve:
  >
  > - Billing account 0149F8-DB6353-622382
  > - Google AI Studio Tier 3
  > - Existing project hierarchy
  >
  > Do not modify billing account 01E88E-11F37C-ADCBD5.
  >
  > Please verify that no IAM deny policy or Principal Access Boundary blocks these permissions.
  >
  > Thank you,
  >
  > Aaryn Biro
  > aaryn.biro@xcelsior 

  Append this:

   > Additional verification: generativelanguage.googleapis.com is currently disabled on digital-circuit-hkw4x; gcloud services list --enabled returned “Listed 0 items.”
  >
  > Please enable it and grant the four temporary project-level roles previously requested:
  >
  > - roles/serviceusage.serviceUsageAdmin
  > - roles/serviceusage.apiKeysAdmin
  > - roles/iam.serviceAccountCreator
  > - roles/iam.serviceAccountApiKeyBindingAdmin


  • Send it through both official channels:

  1. Email cloudstartupsupport@google.com (mailto:cloudstartupsupport@google.com) from aaryn.biro@xcelsior.ca. Google lists this as its official Startup Program contact. Startup FAQ
     (https://cloud.google.com/startup/faq)

     Subject:
     Tier 3 project inaccessible in Google-managed organization — digital-circuit-hkw4x

  2. Open a Cloud Billing Support case (https://console.cloud.google.com/support/billing):
      - Click Get billing support
      - Select Billing
      - Choose billing account 0149F8-DB6353-622382 if available
      - Category: Account/project access or Gemini/AI Studio
      - Paste the same complete message

  Do not select or request changes to credit account 01E88E-11F37C-ADCBD5. Google confirms billing support is free for all Cloud accounts. Billing support instructions
  (https://docs.cloud.google.com/support/docs/get-billing-support)

  Attach screenshots or text showing:

  - AI Studio Tier 3 status
  - The missing-permissions errors
  - testIamPermissions results
  - Disabled Generative Language API result
  - Full ancestor hierarchy

  If 0149F8-DB6353-622382 cannot be selected, use the Support Contact Troubleshooter linked on the billing-support page and choose lost access to a project/billing account.

   Correct — Google One AI Pro is NOT a Vertex AI Tier. Google treats consumer subscriptions and enterprise cloud infrastructure as two
  totally different worlds.

  Here is the straightforward breakdown of what you actually have, and how to get maximum throughput immediately:
  ──────
  ### What You Currently Have

   Subscription / Account         | What It Is                             | Actual Rate Limits / Tiers
  --------------------------------|----------------------------------------|-----------------------------------------------------------
   Google One AI Pro (@gmail.com) | Consumer web & IDE subscription        | High daily usage for Android Studio & Gemini Advanced web
                                  | ($20/mo)                               | chat. 0 GCP / Vertex AI credits.
   Google AI Studio (Free Tier)   | Developer testing API keys             | Capped at 15 RPM (Flash) and 2–5 RPM (Pro models).
   Vertex AI (xcelsior-gpu)       | Corporate GCP project (@xcelsior.ca)   | Standard Gemini limits (~300 RPM). Blocked on Claude by
                                  |                                        | NOT_ENOUGH_USAGE_HISTORY.
  ──────
  ### How to Get "Tier 3 / High Throughput" Right Now

  Depending on whether you want high-throughput Gemini API or Claude API, here is how you get it immediately:

  #### 1. To get Tier 3 throughput for Gemini API:

  • Action: Enable Pay-As-You-Go Billing in Google AI Studio https://aistudio.google.com/ on your @gmail.com account.
  • Result: Your API key instantly upgrades from the 15 RPM Free Tier to Paid Tiers offering 1,000 – 2,000+ RPM and 4,000,000+ TPM. You
  only pay for what you actually consume.

  #### 2. To get high-throughput Claude on Vertex AI (xcelsior-gpu):

  • Action: Submit a manual quota ticket in GCP to bypass NOT_ENOUGH_USAGE_HISTORY.
  • Path: Go to GCP Console → Support → Create Case → Vertex AI Quota Increase for project 152103501471 (xcelsior-gpu).
  • Request: Ask for a quota unblock for anthropic-claude-3-5-sonnet / claude-3-7-sonnet for production enterprise use. Manual review
  bypasses the automated rejection.


• Subject: Google for Startups project migration assistance

  Hello Google Cloud Startups Support,

  My approved Startup Program billing account is:

  - Billing account: Xcelsior Cloud Billing
  - Billing ID: 01E88E-11F37C-ADCBD5
  - Destination organization: xcelsior.ca (812964255343)
  - Migration identity: aaryn.biro@xcelsior.ca

  I need to migrate:

  - Project: digital-circuit-hkw4x
  - Project number: 480880453449
  - Source organization: 71994629047
  - Source parent folder: 21358131809

  My destination organization is ready, but I cannot complete the source-side requirements. Please:

  1. Allow export from source organization 71994629047 to under:organizations/812964255343.
  2. Grant roles/resourcemanager.projectMover to aaryn.biro@xcelsior.ca on folder 21358131809.
  3. Grant roles/resourcemanager.projectIamAdmin and roles/billing.projectManager to the same identity on the project.

  The project currently uses billing account 0149F8-DB6353-622382, which I cannot access. After migration, it needs to use my approved Startup billing account 01E88E-11F37C-ADCBD5.

  If this project is managed and cannot be exported, please confirm and advise how its resources should be transferred to a project under my organization.

  Thank you,
  Aaryn Biro
  aaryn.biro@xcelsior.ca


Hello Google for Startups and Google Cloud Billing Support,

  My Google AI Studio account shows the following imported project
  as Gemini API Tier 3:

  - Project ID: digital-circuit-hkw4x
  - Project number: 480880453449
  - Billing account: 0149F8-DB6353-622382
  - Displayed Gemini tier: Tier 3

  However, the Tier 3 entitlement is currently unusable.

  My primary development identity, aaryn.biro@xcelsior.ca, can
  discover the project and has:

  - resourcemanager.projects.get
  - serviceusage.services.use

  But it cannot:

  - Enable generativelanguage.googleapis.com
  - List or create API keys
  - View or modify the project IAM policy
  - Administer the associated billing account

  The specific failures are missing serviceusage.services.enable,
  apikeys.keys.list, and related administrative permissions.

  My consumer account, aaryn.alexander@gmail.com, has Google AI Pro
  but does not have Cloud access to digital-circuit-hkw4x. Logging
  into that account therefore does not resolve the problem.

  My active development projects—including xcelsior-gpu, xcelsior-
  502014, pixelenhance-labs, pixelspark-502414, and phantom-trades-
  mvp—are attached to a different billing account and currently show
  Tier 1. Importing these projects into AI Studio did not transfer
  or expose the Tier 3 entitlement.

  Please help with one of the following resolutions:

  1. Restore full administrative access to digital-circuit-hkw4x for
     aaryn.biro@xcelsior.ca, including permission to enable the
     Gemini API and securely create/restrict API keys; or

  2. Transfer the Tier 3 entitlement and eligible startup/trial
     billing relationship to xcelsior-gpu; or

  3. Provide the correct owner/admin identity and recovery procedure
     for digital-circuit-hkw4x.

  Please preserve the existing Tier 3 status during any ownership,
  IAM, billing-account, or project migration.

  We also need confirmation that the resolved Tier 3 access covers
  the current development models and quotas, particularly:

  - Gemini 3.6 Flash
  - Gemini 3.5 Flash and Flash-Lite
  - Gemini 3.1 Pro
  - Gemini image-generation models
  - Gemini Omni Flash
  - Veo 3.1 Generate and Fast, including native video extension

  Please also confirm:

  - Which startup or trial credits are attached to billing account
    0149F8-DB6353-622382

  - Whether those credits can be used for Gemini API and Veo charges
  - Whether moving the project or billing relationship would reduce
    its Tier 3 status

  - The current Tier 3 RPM, TPM, daily video-generation, and spend
    limits

  - Whether a startup-program representative must authorize the
    transfer

  No paid model, image, or video generation calls were made while
  diagnosing this issue.

  Thank you,

  Aaryn Biro
  aaryn.biro@xcelsior.ca
  Google AI Pro: aaryn.alexander@gmail.com

  Hello Google for Startups and Google Cloud Billing Support,

  My Google AI Studio account shows the following imported project
  as Gemini API Tier 3:

  - Project ID: digital-circuit-hkw4x
  - Project number: 480880453449
  - Billing account: 0149F8-DB6353-622382
  - Displayed Gemini tier: Tier 3

  However, the Tier 3 entitlement is currently unusable.

  My primary development identity, aaryn.biro@xcelsior.ca, can
  discover the project and has:

  - resourcemanager.projects.get
  - serviceusage.services.use

  But it cannot:

  - Enable generativelanguage.googleapis.com
  - List or create API keys
  - View or modify the project IAM policy
  - Administer the associated billing account

  The specific failures are missing serviceusage.services.enable,
  apikeys.keys.list, and related administrative permissions.

  My consumer account, aaryn.alexander@gmail.com, has Google AI Pro
  but does not have Cloud access to digital-circuit-hkw4x. Logging
  into that account therefore does not resolve the problem.

  My active development projects—including xcelsior-gpu, xcelsior-
  502014, pixelenhance-labs, pixelspark-502414, and phantom-trades-
  mvp—are attached to a different billing account and currently show
  Tier 1. Importing these projects into AI Studio did not transfer
  or expose the Tier 3 entitlement.

  Please help with one of the following resolutions:

  1. Restore full administrative access to digital-circuit-hkw4x for
     aaryn.biro@xcelsior.ca, including permission to enable the
     Gemini API and securely create/restrict API keys; or

  2. Transfer the Tier 3 entitlement and eligible startup/trial
     billing relationship to xcelsior-gpu; or

  3. Provide the correct owner/admin identity and recovery procedure
     for digital-circuit-hkw4x.

  Please preserve the existing Tier 3 status during any ownership,
  IAM, billing-account, or project migration.

  We also need confirmation that the resolved Tier 3 access covers
  the current development models and quotas, particularly:

  - Gemini 3.6 Flash
  - Gemini 3.5 Flash and Flash-Lite
  - Gemini 3.1 Pro
  - Gemini image-generation models
  - Gemini Omni Flash
  - Veo 3.1 Generate and Fast, including native video extension

  Please also confirm:

  - Which startup or trial credits are attached to billing account
    0149F8-DB6353-622382

  - Whether those credits can be used for Gemini API and Veo charges
  - Whether moving the project or billing relationship would reduce
    its Tier 3 status

  - The current Tier 3 RPM, TPM, daily video-generation, and spend
    limits

  - Whether a startup-program representative must authorize the
    transfer

  No paid model, image, or video generation calls were made while
  diagnosing this issue.

  Thank you,

  Aaryn Biro
  aaryn.biro@xcelsior.ca
  Google AI Pro: aaryn.alexander@gmail.com