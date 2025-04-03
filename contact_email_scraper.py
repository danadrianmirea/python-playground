import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urlparse

def extract_emails(url):
    try:
        # Add http:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        # Get domain name
        domain = urlparse(url).netloc
        
        # Make request and get content
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all text content
        text_content = soup.get_text()
        
        # Regular expression for email
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text_content)
        
        # Save results
        with open('email_results.txt', 'a') as f:
            for email in emails:
                f.write(f"Website: {domain}\nEmail: {email}\n\n")
                
        print(f"Found {len(emails)} email(s) from {domain}")
        return emails
        
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return []

# Example usage
if __name__ == "__main__":
    websites = [
        "example.com",
        "example.org"
        # Add more websites here
    ]
    
    for site in websites:
        extract_emails(site)
