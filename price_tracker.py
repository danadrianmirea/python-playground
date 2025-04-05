import requests
from bs4 import BeautifulSoup
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime
import json
import os
from urllib.parse import urlparse

TIME_BETWEEN_CHECKS = 100

class PriceTracker:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.products_file = 'products.json'
        self.price_history_file = 'price_history.csv'
        self.timeout = 100  # Timeout constant in seconds
        self.load_products()
        self.setup_price_history()

    def load_products(self):
        """Load tracked products from JSON file"""
        if os.path.exists(self.products_file):
            with open(self.products_file, 'r') as f:
                self.products = json.load(f)
                # Add name field to existing products if missing
                for product in self.products:
                    if 'name' not in product:
                        parsed_url = urlparse(product['url'])
                        path_parts = parsed_url.path.strip('/').split('/')
                        product['name'] = path_parts[-1].replace('-', ' ').title() if path_parts else "Unknown Product"
                self.save_products()  # Save the updated products with names
        else:
            self.products = []
            self.save_products()

    def save_products(self):
        """Save tracked products to JSON file"""
        with open(self.products_file, 'w') as f:
            json.dump(self.products, f, indent=4)

    def setup_price_history(self):
        """Initialize or load price history CSV file"""
        if not os.path.exists(self.price_history_file):
            df = pd.DataFrame(columns=['product_name', 'url', 'price', 'timestamp'])
            df.to_csv(self.price_history_file, index=False)
        self.price_history = pd.read_csv(self.price_history_file)

    def add_product(self, url, target_price, timeout=100):
        """Add a new product to track"""
        # Validate URL
        if not self.is_valid_url(url):
            print("Error: Invalid URL format")
            return False

        # Extract product name from URL
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        product_name = path_parts[-1].replace('-', ' ').title() if path_parts else "Unknown Product"

        product = {
            'url': url,
            'target_price': float(target_price),
            'store': self.detect_store(url),
            'timeout': timeout,
            'name': product_name
        }
        self.products.append(product)
        self.save_products()
        print(f"Added product '{product_name}' from {product['store']} to tracking list")
        return True

    def is_valid_url(self, url):
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def detect_store(self, url):
        """Detect which store the URL belongs to"""
        domain = urlparse(url).netloc.lower()
        if 'amazon.com' in domain or 'amazon.co.uk' in domain:
            return 'amazon'
        elif 'emag.ro' in domain:
            return 'emag'
        else:
            raise ValueError(f"Unsupported store: {domain}")

    def get_amazon_price(self, url):
        """Extract price and name from Amazon product page"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract product name
            name_element = soup.select_one('#productTitle')
            product_name = name_element.text.strip() if name_element else None
            
            # Try different price selectors
            price_element = soup.select_one('.a-price-whole')
            if price_element:
                price = float(price_element.text.replace(',', ''))
                return price, product_name
            
            # Alternative price selector
            price_element = soup.select_one('#priceblock_ourprice')
            if price_element:
                price = float(price_element.text.replace('$', '').replace(',', ''))
                return price, product_name
            
            return None, product_name
        except Exception as e:
            print(f"Error getting Amazon price: {str(e)}")
            return None, None

    def get_emag_price(self, url):
        """Extract price and name from eMAG product page"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract product name
            name_element = soup.select_one('.page-title')
            product_name = name_element.text.strip() if name_element else None
            
            # Try different price selectors for eMAG
            price_element = soup.select_one('.product-new-price')
            if price_element:
                # eMAG prices are in format "1.234,56 Lei"
                price_text = price_element.text.strip().replace('Lei', '').replace('.', '').replace(',', '.')
                price = float(price_text)
                return price, product_name
            
            # Alternative price selector
            price_element = soup.select_one('.price')
            if price_element:
                price_text = price_element.text.strip().replace('Lei', '').replace('.', '').replace(',', '.')
                price = float(price_text)
                return price, product_name
            
            return None, product_name
        except Exception as e:
            print(f"Error getting eMAG price: {str(e)}")
            return None, None

    def get_product_price(self, product):
        """Get price and name based on store type"""
        if product['store'] == 'amazon':
            return self.get_amazon_price(product['url'])
        elif product['store'] == 'emag':
            return self.get_emag_price(product['url'])
        else:
            print(f"Unsupported store: {product['store']}")
            return None, None

    def check_prices(self):
        """Check prices for all tracked products"""
        for product in self.products:
            current_price, website_name = self.get_product_price(product)
            
            if current_price:
                # Update product name if we got a better one from the website
                if website_name:
                    product['name'] = website_name
                    self.save_products()
                
                # Record price in history
                new_row = {
                    'product_name': product['name'],
                    'url': product['url'],
                    'price': current_price,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.price_history = pd.concat([self.price_history, pd.DataFrame([new_row])], ignore_index=True)
                self.price_history.to_csv(self.price_history_file, index=False)

                # Check if price is below target
                if current_price <= product['target_price']:
                    #self.send_price_alert(product, current_price)
                    print(f"Price has changed for {product['name']} to {current_price}")
                
                print(f"{product['name']}: Current price: ${current_price:.2f}")
            
            # Sleep for the product's timeout before checking the next product
            time.sleep(product['timeout'])

    def send_price_alert(self, product, current_price):
        """Send email alert when price drops below target"""
        # Note: You'll need to configure your email settings
        sender_email = "your_email@example.com"
        receiver_email = "your_email@example.com"
        password = "your_app_password"  # Use app-specific password for Gmail

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"Price Alert: {product['name']}"

        body = f"""
        The price for {product['name']} has dropped to ${current_price:.2f}!
        Target price was: ${product['target_price']:.2f}
        Store: {product['store'].upper()}
        Product URL: {product['url']}
        """
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            server.quit()
            print(f"Price alert sent for {product['name']}")
        except Exception as e:
            print(f"Error sending email: {str(e)}")

def main():
    tracker = PriceTracker()
    
    # Example usage
    if not tracker.products:
        #tracker.add_product("https://www.amazon.com/example-product")        
        tracker.add_product("https://www.emag.ro/placa-video-nvidia-geforce-gtx-1050ti-4gb-gddr5-128bit-nvidiagtx1050ti/pd/DNJX06YBM/", 3)
    
    while True:
        print("\nChecking prices...")
        tracker.check_prices()
        print(f"\nWaiting for {TIME_BETWEEN_CHECKS} seconds before next check...")
        time.sleep(TIME_BETWEEN_CHECKS)

if __name__ == "__main__":
    main() 