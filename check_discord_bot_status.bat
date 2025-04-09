@echo off
echo Checking Discord Bot status on Google Cloud Run...

REM Set your project ID
set PROJECT_ID=eddies-discord-bot

REM Check the service status
gcloud run services describe discord-bot --region us-central1 --project %PROJECT_ID%

REM Check the logs
echo.
echo Recent logs:
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=discord-bot" --limit 10 --project %PROJECT_ID%

echo.
echo Status check complete! 