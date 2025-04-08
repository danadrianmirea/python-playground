import os
import json
import logging
import requests
from flask import Flask, request, Response
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# WhatsApp API credentials
WHATSAPP_TOKEN = os.environ.get('WHATSAPP_TOKEN')
VERIFY_TOKEN = os.environ.get('VERIFY_TOKEN')

# WhatsApp API endpoint
WHATSAPP_API_URL = "https://graph.facebook.com/v17.0"

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """
    Verify webhook endpoint for WhatsApp API.
    This is used during the initial setup of the webhook.
    """
    # Parse params from the webhook verification request
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    # Check if a token and mode were sent
    if mode and token:
        # Check the mode and token sent are correct
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            # Respond with 200 OK and the challenge token from the request
            logger.info("Webhook verified!")
            return Response(challenge, status=200)
        else:
            # Responds with '403 Forbidden' if verify tokens do not match
            logger.error("Webhook verification failed!")
            return Response(status=403)
    
    return Response(status=400)

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Handle incoming messages from WhatsApp.
    This is the main webhook that receives messages and sends responses.
    """
    # Parse the request body from the POST
    body = request.get_json()
    
    # Check if this is a WhatsApp API event
    if body.get('object'):
        if (
            body.get('entry') and 
            body['entry'][0].get('changes') and 
            body['entry'][0]['changes'][0].get('value') and 
            body['entry'][0]['changes'][0]['value'].get('messages') and 
            body['entry'][0]['changes'][0]['value']['messages'][0]
        ):
            # Get the phone number and message
            phone_number_id = body['entry'][0]['changes'][0]['value']['metadata']['phone_number_id']
            from_number = body['entry'][0]['changes'][0]['value']['messages'][0]['from']
            msg_body = body['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
            
            logger.info(f"Received message from {from_number}: {msg_body}")
            
            # Echo the message back
            send_message(phone_number_id, from_number, msg_body)
            
            return Response(status=200)
    
    # Return a '404 Not Found' if event is not recognized
    return Response(status=404)

def send_message(phone_number_id, to_number, message):
    """
    Send a message to a WhatsApp user.
    
    Args:
        phone_number_id: The WhatsApp phone number ID
        to_number: The recipient's phone number
        message: The message to send
    """
    url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info(f"Message sent successfully to {to_number}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending message: {str(e)}")

if __name__ == "__main__":
    # Check if required environment variables are set
    if not WHATSAPP_TOKEN or not VERIFY_TOKEN:
        logger.error("Missing required environment variables. Please set WHATSAPP_TOKEN and VERIFY_TOKEN.")
        exit(1)
    
    # Run the Flask app
    app.run(debug=True, port=5000) 