import time, random
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import *
import tkinter as tk
from tkinter import messagebox
from cryptography.fernet import Fernet
import base64
import os
import hashlib
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

minSleepTime = 2
maxSleepTime = 2

def prRed(prt):
    print(f"\033[91m{prt}\033[00m")

def prGreen(prt):
    print(f"\033[92m{prt}\033[00m")

def prYellow(prt):
    print(f"\033[93m{prt}\033[00m")

def get_encryption_key():
    hostname = os.getenv('COMPUTERNAME', 'default')
    key = hashlib.sha256(hostname.encode()).digest()
    return base64.urlsafe_b64encode(key)

def read_settings(filename, required=False):
    settings = {}
    try:
        # Check for encrypted version first
        encrypted_file = filename + '.encrypted'
        if os.path.exists(encrypted_file):
            return decrypt_file(encrypted_file)
            
        # If no encrypted version, read normal file
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        settings[key.strip()] = value.strip()
                        
        # If this is password.txt, encrypt it
        if filename == 'password.txt' and settings:
            encrypt_file(filename)
            
    except FileNotFoundError:
        if required:
            prRed(f"Error: {filename} file not found!")
            exit(1)
        else:
            prYellow(f"Warning: {filename} not found. Some features may be disabled.")
    except Exception as e:
        prRed(f"Error reading {filename}: {str(e)}")
        exit(1)
    return settings

def encrypt_file(filename):
    key = get_encryption_key()
    f = Fernet(key)
    
    with open(filename, 'rb') as file:
        data = file.read()
    
    encrypted_data = f.encrypt(data)
    
    with open(filename + '.encrypted', 'wb') as file:
        file.write(encrypted_data)
    
    os.remove(filename)
    prGreen(f"Created encrypted file: {filename}.encrypted")

def decrypt_file(filename):
    key = get_encryption_key()
    f = Fernet(key)
    
    try:
        with open(filename, 'rb') as file:
            encrypted_data = file.read()
        
        decrypted_data = f.decrypt(encrypted_data)
        
        settings = {}
        for line in decrypted_data.decode().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    settings[key.strip()] = value.strip()
        return settings
    except Exception as e:
        prRed(f"Error decrypting file: {str(e)}")
        return {}

def chromeBrowserOptions():
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-extensions")
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--enable-unsafe-swiftshader')
    options.add_argument("--log-level=3")  
    options.add_argument("--disable-webrtc")
    options.add_argument("--disable-features=WebRTC")
    options.add_argument("--disable-features=WebRtcHideLocalIpsWithMdns")
    options.add_argument("--disable-webassembly")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("--incognito")
    return options

def show_login_popup():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("LinkedIn Login Required", 
                       "Please log in to LinkedIn in the browser window.\n\n"
                       "After successful login, press Enter in the terminal to continue.")
    root.destroy()

def get_profile_info(driver, name):
    try:
        # Search for the person
        search_url = f"https://www.linkedin.com/search/results/people/?keywords={name.replace(' ', '%20')}"
        driver.get(search_url)
        
        try:
            # Wait for the search results container to be present
            results_selector = "div[data-chameleon-result-urn]"
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, results_selector))
            )
            
            # Wait for the page to be fully loaded (no loading spinners)
            WebDriverWait(driver, 15).until_not(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.artdeco-loader"))
            )
            
            search_results = driver.find_elements(By.CSS_SELECTOR, results_selector)
            if not search_results:
                prRed(f"No results found for {name} even after waiting.")
                return f"{name}, Error: No search results found"

            # Get the first result
            first_result = search_results[0]
            
            # Extract profession directly from search result
            try:
                # Wait for the profession element to be present
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.pWIDTtuKGUWausIySAtxTfPvqtvw.t-14.t-black.t-normal"))
                )
                profession_element = first_result.find_element(By.CSS_SELECTOR, "div.pWIDTtuKGUWausIySAtxTfPvqtvw.t-14.t-black.t-normal")
                profession = profession_element.text.strip()
                if not profession:
                    profession = "Not found"
            except:
                profession = "Not found"

            return f"{name}, {profession}"

        except Exception as e:
            prRed(f"Error: Search results not loaded for {name} - {str(e)}")
            return f"{name}, Error: Search results not loaded - {str(e)}"

    except Exception as e:
        prRed(f"Error processing {name}: {str(e)}")
        return f"{name}, Error: {str(e)}"

def main():
    # Read credentials
    credentials = read_settings('password.txt', required=False)
    linkedin_username = credentials.get('linkedin_username')
    linkedin_password = credentials.get('linkedin_password')

    # Initialize browser
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), 
                            options=chromeBrowserOptions())
    
    # Login
    driver.get("https://www.linkedin.com/login")
    if linkedin_username and linkedin_password:
        driver.find_element("id","username").send_keys(linkedin_username)
        driver.find_element("id","password").send_keys(linkedin_password)
        driver.find_element("xpath",'//button[@type="submit"]').click()
        time.sleep(maxSleepTime)
    else:
        show_login_popup()
        input("Press Enter to continue...")

    # Read input file
    try:
        with open('input.txt', 'r') as f:
            names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        prRed("Error: input.txt file not found!")
        driver.quit()
        return

    # Process each name
    with open('output.txt', 'a', encoding='utf-8') as f:
        for name in names:
            prYellow(f"Processing: {name}")
            result = get_profile_info(driver, name)
            f.write(result + '\n')
            f.flush()
            prGreen(f"Saved: {result}")

    driver.quit()
    prGreen("Done! Results saved to output.txt")

if __name__ == "__main__":
    main() 