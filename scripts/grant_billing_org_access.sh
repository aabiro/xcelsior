#!/usr/bin/env bash
# Grant full billing + org-view + project access for the three aaryn identities.
# Run as: aaryn.biro@xcelsior.ca (org owner / billing admin)
#
#   gcloud auth login aaryn.biro@xcelsior.ca
#   gcloud auth application-default login
#   bash scripts/grant_billing_org_access.sh
#
set -euo pipefail

BILLING_ACCOUNT="01E88E-11F37C-ADCBD5"
ORG_ID="812964255343"   # xcelsior org (from project parent)
PROJECTS=(
  "xcelsior-502014"
  "pixelenhance-labs"
  "pixelspark-502414"
)

# All three humans who should see billing + org name + projects
USERS=(
  "aaryn.biro@xcelsior.ca"
  "aaryn.alexander@gmail.com"
  "aaryn.biro@pixelenhancelabs.ai"
)

# --- Billing account roles ---
# Admin: full manage (except close sometimes)
# User: link projects / view
# Viewer: read-only spend & account
BILLING_ROLES=(
  "roles/billing.admin"
  "roles/billing.user"
  "roles/billing.viewer"
  "roles/billing.costsManager"   # budgets / cost management UI
)

# --- Organization roles ---
# browser: see org name + resource hierarchy (fixes "Unknown organization")
# org viewer: read org metadata / policies (view only)
# billing creator optional
ORG_ROLES=(
  "roles/resourcemanager.organizationViewer"
  "roles/browser"
  "roles/billing.viewer"   # org-level billing visibility where applicable
)

# --- Project roles ---
# owner keeps full control; editor is alternative if you want less than owner
PROJECT_ROLES=(
  "roles/owner"
  "roles/viewer"
  "roles/serviceusage.serviceUsageViewer"
  "roles/serviceusage.serviceUsageConsumer"
)

echo "Authenticated as:"
gcloud auth list --filter=status:ACTIVE --format='value(account)' || true
echo "Billing: ${BILLING_ACCOUNT}"
echo "Org:     ${ORG_ID}"
echo

for email in "${USERS[@]}"; do
  member="user:${email}"
  echo "======== ${email} ========"

  for role in "${BILLING_ROLES[@]}"; do
    echo "  billing ${role}"
    gcloud billing accounts add-iam-policy-binding "${BILLING_ACCOUNT}" \
      --member="${member}" \
      --role="${role}" \
      --quiet \
      >/dev/null || echo "    WARN: failed billing ${role}"
  done

  for role in "${ORG_ROLES[@]}"; do
    echo "  org ${role}"
    gcloud organizations add-iam-policy-binding "${ORG_ID}" \
      --member="${member}" \
      --role="${role}" \
      --quiet \
      >/dev/null || echo "    WARN: failed org ${role}"
  done

  for proj in "${PROJECTS[@]}"; do
    for role in "${PROJECT_ROLES[@]}"; do
      echo "  project ${proj} ${role}"
      gcloud projects add-iam-policy-binding "${proj}" \
        --member="${member}" \
        --role="${role}" \
        --quiet \
        >/dev/null || echo "    WARN: failed ${proj} ${role}"
    done
  done
  echo
done

echo "======== VERIFY billing IAM ========"
gcloud billing accounts get-iam-policy "${BILLING_ACCOUNT}" \
  --flatten='bindings[].members' \
  --filter='bindings.members:aaryn' \
  --format='table(bindings.role, bindings.members)'

echo
echo "======== VERIFY org IAM (aaryn*) ========"
gcloud organizations get-iam-policy "${ORG_ID}" \
  --flatten='bindings[].members' \
  --filter='bindings.members:aaryn' \
  --format='table(bindings.role, bindings.members)'

echo
echo "Done. Have gmail + pixelenhancelabs users:"
echo "  1) Sign out/in of Cloud Console"
echo "  2) Open https://console.cloud.google.com/billing/${BILLING_ACCOUNT}"
echo "  3) Confirm org name is visible (not Unknown organization)"
echo "  4) Troubleshooter should clear billing.accounts.get / getSpendingInformationScoped"
