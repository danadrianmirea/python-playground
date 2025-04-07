@echo off
echo Copying telegram_bot_gcp.py to main.py...
copy telegram_bot_gcp.py main.py

echo Deploying to Google Cloud Functions...
gcloud functions deploy telegram_webhook --project=telegram-bot-project-456120 --runtime python310 --trigger-http --allow-unauthenticated --entry-point telegram_webhook --set-env-vars TELEGRAM_BOT_TOKEN=7830852520:AAH0rjYJoMIY7PGZd-vo00UYrTUyX2fL154

echo Cleaning up...
del main.py

echo Done!
pause 