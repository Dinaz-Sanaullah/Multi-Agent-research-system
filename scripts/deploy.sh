#!/usr/bin/env bash
# Deploy the ADK research agent to Google Cloud Run
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-multi-agent-research-system}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-research-agent}"
AGENT_PATH="./research_agent"

echo "Deploying to project: $PROJECT_ID, region: $REGION"

# Ensure secrets exist (run once)
# echo "$TAVILY_API_KEY" | gcloud secrets create TAVILY_API_KEY --project=$PROJECT_ID --data-file=- 2>/dev/null || true

adk deploy cloud_run \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --service_name="$SERVICE_NAME" \
  --with_ui=false \
  "$AGENT_PATH"

echo "Deployment complete. Service: $SERVICE_NAME"
