@echo off
echo Copying telegram_bot_gcp.py to main.py...
copy telegram_bot_gcp.py main.py

echo Reading token from token.txt...
set /p TELEGRAM_BOT_TOKEN=<token.txt

echo Deploying to Google Cloud Functions...
REM gcloud functions deploy telegram_webhook --project=telegram-bot-project-456120 --runtime python310 --trigger-http --allow-unauthenticated --entry-point telegram_webhook --set-env-vars TELEGRAM_BOT_TOKEN=7830852520:AAGCl5acH-OV1D_zdf131J3YIJgTdXz3vww
gcloud functions deploy check_reminders --runtime python310 --trigger-topic check-reminders --entry-point check_reminders --project="telegram-bot-project-456120" --region=us-central1 --set-env-vars TELEGRAM_BOT_TOKEN=%TELEGRAM_BOT_TOKEN%

echo Press any key to perform cleanup...
pause
del main.py
echo Done!