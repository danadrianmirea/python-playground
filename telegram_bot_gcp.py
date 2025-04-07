import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import datetime
import pytz
import functions_framework
from flask import Request
import json

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Define command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Test/development bot made by AdrianM. Use /help for a list of commands.')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
Available commands:
/start - Start the bot
/help - Show this help message
/echo <text> - Echo back your text
/time - Show the current time in UTC+0 and Bucharest time
    """
    await update.message.reply_text(help_text)

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Echo the user message."""
    await update.message.reply_text(update.message.text)

async def time_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the current time in UTC+0 and Bucharest time."""
    utc_now = datetime.datetime.now(pytz.UTC)
    bucharest_tz = pytz.timezone('Europe/Bucharest')
    bucharest_now = utc_now.astimezone(bucharest_tz)
    
    time_message = f"Current time:\nUTC+0: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}\nBucharest: {bucharest_now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    await update.message.reply_text(time_message)

async def handle_update(update_dict: dict) -> None:
    """Handle a single update from Telegram."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is not set")
        return

    # Create the Application
    application = Application.builder().token(token).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("echo", echo))
    application.add_handler(CommandHandler("time", time_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Initialize the application
    await application.initialize()
    
    try:
        # Process the update
        update = Update.de_json(update_dict, application.bot)
        await application.process_update(update)
    finally:
        # Clean up
        await application.shutdown()

@functions_framework.http
def telegram_webhook(request: Request):
    """Cloud Function entry point."""
    # Verify the request is from Telegram
    if request.method != "POST":
        return "Only POST requests are accepted", 405

    # Get the update from Telegram
    try:
        update_dict = request.get_json()
        if not update_dict:
            return "Invalid request body", 400
    except json.JSONDecodeError:
        return "Invalid JSON", 400

    # Process the update asynchronously
    import asyncio
    asyncio.run(handle_update(update_dict))
    return "OK", 200 