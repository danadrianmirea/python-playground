import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import datetime
import pytz

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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

async def time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the current time in UTC+0 and Bucharest time."""
    utc_now = datetime.datetime.now(pytz.UTC)
    bucharest_tz = pytz.timezone('Europe/Bucharest')
    bucharest_now = utc_now.astimezone(bucharest_tz)
    
    time_message = f"Current time:\nUTC+0: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}\nBucharest: {bucharest_now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    await update.message.reply_text(time_message)

def main():
    """Start the bot."""
    # Get the bot token from environment variable
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable is not set.")
        print("Please set it with your bot token from BotFather.")
        return

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(token).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("echo", echo))
    application.add_handler(CommandHandler("time", time))
    
    # Add message handler for non-command messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 