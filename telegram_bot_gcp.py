import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import datetime
import pytz
import functions_framework
from flask import Request
import json
from google.cloud import firestore
import re
from dateutil import parser
from dateutil.relativedelta import relativedelta

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Initialize Firestore client
db = firestore.Client()

# Define command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Test/development bot made by AdrianM. Use /help for a list of commands.')
    # Also display the help message
    await help_command(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
Available commands:
/start - Start the bot
/help - Show this help message
/echo <text> - Echo back your text
/time - Show the current time in UTC+0 and Bucharest time
/remind <time> <message> - Set a reminder (e.g., '/remind 5' (minutes), '/remind 30s' (seconds), '/remind 1h' (hours), '/remind 2h30m dinner' or '/remind tomorrow 14:00 meeting')
/reminders - List all your active reminders
/delreminder <id> - Delete a reminder by its ID
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

def parse_time(time_str: str) -> datetime.datetime:
    """Parse time string into datetime object."""
    now = datetime.datetime.now(pytz.UTC)
    
    # Try to parse as a plain number (assume minutes)
    try:
        minutes = int(time_str)
        return now + datetime.timedelta(minutes=minutes)
    except ValueError:
        pass
    
    # Try to parse as relative time (e.g., "2h30m" or "30s")
    relative_pattern = re.compile(r'((?P<hours>\d+)h)?((?P<minutes>\d+)m)?((?P<seconds>\d+)s)?')
    match = relative_pattern.match(time_str)
    
    if match and (match.group('hours') or match.group('minutes') or match.group('seconds')):
        hours = int(match.group('hours') or 0)
        minutes = int(match.group('minutes') or 0)
        seconds = int(match.group('seconds') or 0)
        return now + datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    # Try to parse as absolute time
    try:
        parsed_time = parser.parse(time_str, fuzzy=True)
        if parsed_time.tzinfo is None:
            parsed_time = parsed_time.replace(tzinfo=pytz.UTC)
        
        # If only time was provided (no date), use today's date
        if parsed_time.date() == datetime.datetime.min.date():
            parsed_time = parsed_time.replace(
                year=now.year,
                month=now.month,
                day=now.day
            )
        
        # If the time is in the past, add one day
        if parsed_time < now:
            parsed_time = parsed_time + datetime.timedelta(days=1)
        
        return parsed_time
    except ValueError:
        raise ValueError("Could not parse time. Please use format like '5' (minutes), '30s' (seconds), '2h30m' or 'tomorrow 14:00'")

async def remind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set a reminder."""
    if not context.args:
        await update.message.reply_text(
            "Please provide both time and message.\n"
            "Examples:\n"
            "- /remind 5 (minutes)\n"
            "- /remind 30s (seconds)\n"
            "- /remind 1h (hours)\n"
            "- /remind 2h30m dinner\n"
            "- /remind tomorrow 14:00 team meeting\n"
            "- /remind 5m check oven"
        )
        return

    try:
        # Check if the message is in quotes
        full_text = " ".join(context.args)
        quoted_message = None
        
        # Look for quoted text
        if '"' in full_text:
            # Find the first and last quote
            first_quote = full_text.find('"')
            last_quote = full_text.rfind('"')
            
            if first_quote != last_quote:  # We have a complete quote
                # Extract the time part (before the first quote)
                time_part = full_text[:first_quote].strip()
                # Extract the message (between quotes)
                quoted_message = full_text[first_quote+1:last_quote]
                
                # Parse the time
                reminder_time = parse_time(time_part)
                message = quoted_message
            else:
                # Only one quote found, treat as normal
                if len(context.args) == 1:
                    reminder_time = parse_time(context.args[0])
                    message = ""
                else:
                    # Try to parse first two arguments as time
                    time_str = " ".join(context.args[:2])
                    try:
                        reminder_time = parse_time(time_str)
                        message = " ".join(context.args[2:])
                    except ValueError:
                        # If that fails, try just the first argument
                        reminder_time = parse_time(context.args[0])
                        message = " ".join(context.args[1:])
        else:
            # No quotes, process as before
            if len(context.args) == 1:
                reminder_time = parse_time(context.args[0])
                message = ""
            else:
                # Try to parse first two arguments as time
                time_str = " ".join(context.args[:2])
                try:
                    reminder_time = parse_time(time_str)
                    message = " ".join(context.args[2:])
                except ValueError:
                    # If that fails, try just the first argument
                    reminder_time = parse_time(context.args[0])
                    message = " ".join(context.args[1:])

        # Store reminder in Firestore
        reminder_ref = db.collection('reminders').document()
        reminder = {
            'user_id': update.effective_user.id,
            'chat_id': update.effective_chat.id,
            'time': reminder_time,
            'message': message,
            'created_at': datetime.datetime.now(pytz.UTC)
        }
        reminder_ref.set(reminder)

        # Format confirmation message
        time_str = reminder_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        await update.message.reply_text(
            f"✅ Reminder set!\n"
            f"⏰ Time: {time_str}\n"
            f"📝 Message: {message}\n"
            f"🆔 ID: {reminder_ref.id}"
        )

    except ValueError as e:
        await update.message.reply_text(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error setting reminder: {str(e)}")
        await update.message.reply_text("Sorry, something went wrong while setting the reminder.")

async def list_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all active reminders for the user."""
    try:
        # Query reminders from Firestore
        reminders = db.collection('reminders')\
            .where('user_id', '==', update.effective_user.id)\
            .where('time', '>', datetime.datetime.now(pytz.UTC))\
            .order_by('time')\
            .stream()
        
        # Format reminders list
        reminder_list = []
        for reminder in reminders:
            data = reminder.to_dict()
            time_str = data['time'].strftime('%Y-%m-%d %H:%M:%S %Z')
            reminder_list.append(
                f"🆔 {reminder.id}\n"
                f"⏰ {time_str}\n"
                f"📝 {data['message']}\n"
            )
        
        if reminder_list:
            message = "Your active reminders:\n\n" + "\n".join(reminder_list)
        else:
            message = "You have no active reminders."
        
        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error listing reminders: {str(e)}")
        await update.message.reply_text("Sorry, something went wrong while fetching your reminders.")

async def delete_reminder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Delete a reminder by its ID."""
    if not context.args:
        await update.message.reply_text("Please provide the reminder ID to delete.")
        return

    reminder_id = context.args[0]
    try:
        # Get the reminder and verify ownership
        reminder_ref = db.collection('reminders').document(reminder_id)
        reminder = reminder_ref.get()
        
        if not reminder.exists:
            await update.message.reply_text("❌ Reminder not found.")
            return
        
        if reminder.to_dict()['user_id'] != update.effective_user.id:
            await update.message.reply_text("❌ You can only delete your own reminders.")
            return
        
        # Delete the reminder
        reminder_ref.delete()
        await update.message.reply_text("✅ Reminder deleted successfully!")

    except Exception as e:
        logger.error(f"Error deleting reminder: {str(e)}")
        await update.message.reply_text("Sorry, something went wrong while deleting the reminder.")

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
    application.add_handler(CommandHandler("remind", remind))
    application.add_handler(CommandHandler("reminders", list_reminders))
    application.add_handler(CommandHandler("delreminder", delete_reminder))
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