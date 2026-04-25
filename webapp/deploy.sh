2#!/bin/bash
# ============================================================
# HDB Knowledge Base — Deploy to S3 + CloudFront
# Usage: ./deploy.sh <s3-bucket-name> <cloudfront-distribution-id>
#
# Example:
#   ./deploy.sh my-hdb-kb-bucket E1ABCDEFGHIJKL
#
# Requires: AWS CLI configured (aws configure)
# ============================================================

set -e

BUCKET="${1}"
CF_DIST="${2}"

# ── Validate arguments ──────────────────────────────────────
if [ -z "$BUCKET" ] || [ -z "$CF_DIST" ]; then
  echo "Usage: $0 <s3-bucket-name> <cloudfront-distribution-id>"
  echo "Example: $0 my-hdb-kb-bucket E1ABCDEFGHIJKL"
  exit 1
fi

# ── Check AWS CLI ───────────────────────────────────────────
if ! command -v aws &>/dev/null; then
  echo "Error: AWS CLI not found. Install from https://aws.amazon.com/cli/"
  exit 1
fi

echo "=================================================="
echo "  HDB KB Deploy → s3://$BUCKET"
echo "=================================================="

# ── Step 1: Build search index ─────────────────────────────
echo ""
echo "[1/3] Building search index..."
python3 "$(dirname "$0")/build_index.py"

# ── Step 2: Upload all webapp files ────────────────────────
echo ""
echo "[2/3] Uploading webapp/ to s3://$BUCKET/..."
aws s3 sync "$(dirname "$0")/" "s3://$BUCKET/" \
  --exclude "deploy.sh" \
  --exclude "build_index.py" \
  --exclude ".DS_Store" \
  --delete \
  --cache-control "max-age=300"

# ── Step 3: CloudFront cache invalidation ───────────────────
echo ""
echo "[3/3] Invalidating CloudFront cache..."
aws cloudfront create-invalidation \
  --distribution-id "$CF_DIST" \
  --paths "/*" \
  --output text --query 'Invalidation.Id'

echo ""
echo "✅ Deploy complete!"
echo "   Your KB is live at your CloudFront domain."
echo ""
echo "   To update after a new submission:"
echo "   1. Update webapp/submission_journey.html"
echo "   2. Run this script again"
echo "   3. Also update BEST_SCORE in webapp/nav.js if score improved"
