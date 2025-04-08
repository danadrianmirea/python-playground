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
try:
    db = firestore.Client()
    logger.info("Firestore client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firestore client: {str(e)}")
    # We'll still set db to None so we can check later
    db = None

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
/time - Show the current time in UTC+0 and Bucharest time
/remind <time> <message> - Set a reminder (e.g., '/remind 5' (minutes), '/remind 1h' (hours), '/remind 2h30m dinner' or '/remind tomorrow 14:00 meeting')
/reminders - List all your active reminders
/delreminder <id> - Delete a reminder by its ID
    """
    await update.message.reply_text(help_text)

async def invalid_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle invalid commands by prefixing the message with 'Invalid command: '."""
    await update.message.reply_text(f"Invalid command: {update.message.text}")

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
    
    # Try to parse as relative time (e.g., "2h30m")
    relative_pattern = re.compile(r'((?P<hours>\d+)h)?((?P<minutes>\d+)m)?')
    match = relative_pattern.match(time_str)
    
    if match and (match.group('hours') or match.group('minutes')):
        hours = int(match.group('hours') or 0)
        minutes = int(match.group('minutes') or 0)
        return now + datetime.timedelta(hours=hours, minutes=minutes)
    
    # Try to parse as absolute time
    try:
        # Split the string to handle cases where a message might be included
        parts = time_str.split()
        time_part = parts[0]  # Only use the first part for time parsing
        
        parsed_time = parser.parse(time_part, fuzzy=True)
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
        raise ValueError("Could not parse time. Please use format like '5' (minutes), '2h30m' or 'tomorrow 14:00'")

async def remind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set a reminder."""
    if not context.args:
        await update.message.reply_text(
            "Please provide both time and message.\n"
            "Examples:\n"
            "- /remind 5 (minutes)\n"
            "- /remind 1h (hours)\n"
            "- /remind 2h30m dinner\n"
            "- /remind tomorrow 14:00 team meeting\n"
            "- /remind 5m check oven"
        )
        return

    try:
        # Check if Firestore client is initialized
        if db is None:
            logger.error("Firestore client is not initialized")
            await update.message.reply_text("Database connection error. Please try again later.")
            return
            
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
                    # Try to parse first argument as time
                    reminder_time = parse_time(context.args[0])
                    message = " ".join(context.args[1:])
        else:
            # No quotes, process as before
            if len(context.args) == 1:
                reminder_time = parse_time(context.args[0])
                message = ""
            else:
                # Try to parse first argument as time
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
        # Check if Firestore client is initialized
        if db is None:
            logger.error("Firestore client is not initialized")
            await update.message.reply_text("Database connection error. Please try again later.")
            return
            
        logger.info(f"Fetching reminders for user {update.effective_user.id}")
        
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
        
        logger.info(f"Found {len(reminder_list)} reminders for user {update.effective_user.id}")
        
        if reminder_list:
            message = "Your active reminders:\n\n" + "\n".join(reminder_list)
        else:
            message = "You have no active reminders."
        
        await update.message.reply_text(message)

    except Exception as e:
        error_message = f"Error listing reminders: {str(e)}"
        logger.error(error_message)
        # Send a more detailed error message to help with debugging
        await update.message.reply_text(f"Error fetching reminders: {str(e)}")

async def delete_reminder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Delete a reminder by its ID."""
    if not context.args:
        await update.message.reply_text("Please provide the reminder ID to delete.")
        return

    reminder_id = context.args[0]
    try:
        # Check if Firestore client is initialized
        if db is None:
            logger.error("Firestore client is not initialized")
            await update.message.reply_text("Database connection error. Please try again later.")
            return
            
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
        await update.message.reply_text(f"Error deleting reminder: {str(e)}")

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
    application.add_handler(CommandHandler("time", time_command))
    application.add_handler(CommandHandler("remind", remind))
    application.add_handler(CommandHandler("reminders", list_reminders))
    application.add_handler(CommandHandler("delreminder", delete_reminder))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, invalid_command))

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

@functions_framework.cloud_event
def check_reminders(cloud_event):
    """Cloud Function that runs on a schedule to check for expired reminders."""
    logger.info("Checking for expired reminders")
    
    try:
        # Check if Firestore client is initialized
        if db is None:
            logger.error("Firestore client is not initialized")
            return
            
        # Get current time
        now = datetime.datetime.now(pytz.UTC)
        
        # Query for reminders that have expired (time <= now)
        expired_reminders = db.collection('reminders')\
            .where('time', '<=', now)\
            .stream()
        
        # Process each expired reminder
        for reminder in expired_reminders:
            data = reminder.to_dict()
            user_id = data.get('user_id')
            chat_id = data.get('chat_id')
            message = data.get('message', '')
            
            if not user_id or not chat_id:
                logger.error(f"Reminder {reminder.id} missing user_id or chat_id")
                continue
                
            # Send notification to user and only delete if successful
            if send_reminder_notification(user_id, chat_id, message, reminder.id):
                reminder.reference.delete()
                logger.info(f"Deleted expired reminder {reminder.id}")
            else:
                logger.error(f"Failed to send notification for reminder {reminder.id}, will retry next time")
            
    except Exception as e:
        logger.error(f"Error checking reminders: {str(e)}")

def send_reminder_notification(user_id, chat_id, message, reminder_id):
    """Send a notification to a user about an expired reminder.
    Returns True if notification was sent successfully, False otherwise."""
    try:
        # Get the bot token
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        if not token:
            logger.error("TELEGRAM_BOT_TOKEN environment variable is not set")
            return False
            
        # Create a simple bot instance
        from telegram import Bot
        bot = Bot(token=token)
        
        # Send the notification
        notification_text = f"⏰ REMINDER!\n\n"
        if message:
            notification_text += f"📝 {message}\n\n"
        notification_text += f"🆔 ID: {reminder_id}"
        
        # Use asyncio to send the message
        import asyncio
        asyncio.run(bot.send_message(chat_id=chat_id, text=notification_text))
        logger.info(f"Sent reminder notification to user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending reminder notification: {str(e)}")
        return False 