import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import datetime
import pytz
import requests
import random

# Load environment variables from .env file
load_dotenv()

# Get the Discord token from environment variables
TOKEN = os.getenv('DISCORD_TOKEN')

# Set up the bot with intents
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    """Event that runs when the bot is ready."""
    print(f'{bot.user.name} has connected to Discord!')
    print(f'Bot is in {len(bot.guilds)} guilds')
    print('Bot is ready to use! Available commands:')
    print('  !ping - Bot responds with Pong!')
    print('  !echo <message> - Bot repeats your message')
    print('  !time - Show the current time in UTC+0 and Bucharest time')
    print('  !joke - Get a random joke')
    print('  !meme - Get a random meme')

@bot.command(name='ping')
async def ping(ctx):
    """Simple command that responds with Pong!"""
    await ctx.send('Pong!')

@bot.command(name='echo')
async def echo(ctx, *, message):
    """Echo the message back to the channel."""
    await ctx.send(message)

@bot.command(name='time')
async def time(ctx):
    """Show the current time in UTC+0 and Bucharest time."""
    utc_now = datetime.datetime.now(pytz.UTC)
    bucharest_tz = pytz.timezone('Europe/Bucharest')
    bucharest_now = utc_now.astimezone(bucharest_tz)
    
    time_message = f"Current time:\nUTC+0: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}\nBucharest: {bucharest_now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    await ctx.send(time_message)

@bot.command(name='joke')
async def joke(ctx):
    """Send a random joke when the command !joke is issued."""
    try:
        # Using the JokeAPI to get a random joke
        response = requests.get("https://v2.jokeapi.dev/joke/Any?safe-mode")
        if response.status_code == 200:
            joke_data = response.json()
            
            if joke_data["type"] == "single":
                joke_text = joke_data["joke"]
            else:
                joke_text = f"{joke_data['setup']}\n\n{joke_data['delivery']}"
                
            await ctx.send(joke_text)
        else:
            # Fallback jokes in case the API fails
            fallback_jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What do you call a fake noodle? An impasta!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "What do you call a bear with no teeth? A gummy bear!",
                "Why don't skeletons fight each other? They don't have the guts!"
            ]
            await ctx.send(random.choice(fallback_jokes))
    except Exception as e:
        print(f"Error fetching joke: {str(e)}")
        await ctx.send("Sorry, I couldn't fetch a joke right now. Try again later!")

@bot.command(name='meme')
async def meme(ctx):
    """Send a random meme when the command !meme is issued."""
    try:
        # Using the Meme API to get a random meme
        response = requests.get("https://meme-api.com/gimme")
        if response.status_code == 200:
            meme_data = response.json()
            meme_url = meme_data["url"]
            
            # Send the meme image
            embed = discord.Embed(title=meme_data["title"], color=discord.Color.blue())
            embed.set_image(url=meme_url)
            await ctx.send(embed=embed)
        else:
            await ctx.send("Sorry, I couldn't fetch a meme right now. Try again later!")
    except Exception as e:
        print(f"Error fetching meme: {str(e)}")
        await ctx.send("Sorry, I couldn't fetch a meme right now. Try again later!")

@bot.event
async def on_message(message):
    """Event that runs when a message is sent in a channel the bot can see."""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands
    await bot.process_commands(message)

# Run the bot
if __name__ == '__main__':
    if TOKEN:
        try:
            print("Starting bot...")
            print("Available commands:")
            print("  !ping - Bot responds with 'Pong!'")
            print("  !echo <message> - Bot repeats your message")
            print("  !time - Show the current time in UTC+0 and Bucharest time")
            print("  !joke - Get a random joke")
            print("  !meme - Get a random meme")
            bot.run(TOKEN)
        except discord.errors.LoginFailure:
            print("Error: Invalid token. Please check your .env file and make sure the token is correct.")
        except discord.errors.HTTPException as e:
            if e.status == 401:
                print("Error: Unauthorized. Please check your token.")
            elif e.status == 403:
                print("Error: Forbidden. The bot doesn't have the required permissions.")
            else:
                print(f"HTTP Error: {e.status} - {e.text}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    else:
        print("Error: DISCORD_TOKEN not found in environment variables.")
        print("Please create a .env file with your Discord token.") 