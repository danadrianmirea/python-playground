import time,math,random,os
import pickle, hashlib
import logging
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import *
from typing import List
import tkinter as tk
from tkinter import messagebox

def prRed(prt):
    print(f"\033[91m{prt}\033[00m")

def prGreen(prt):
    print(f"\033[92m{prt}\033[00m")

def prYellow(prt):
    print(f"\033[93m{prt}\033[00m")

def read_settings(filename, required=False):
    settings = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        settings[key.strip()] = value.strip()
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

def parse_list_setting(value):
    return [item.strip() for item in value.split(',')]

def parse_bool_setting(value):
    return value.lower() in ['true', '1', 'yes', 'y']

# Read credentials from password.txt (optional)
credentials = read_settings('password.txt', required=False)
linkedin_username = credentials.get('linkedin_username')
linkedin_password = credentials.get('linkedin_password')

# Read all other settings from settings.txt (required)
settings = read_settings('settings.txt', required=True)

# Search configuration
linkedinJobLink = settings.get('linkedin_job_link')

# Search filters
checkTitle = parse_bool_setting(settings.get('check_title', '0'))
checkDescription = parse_bool_setting(settings.get('check_description', '1'))
checkBadDescription = parse_bool_setting(settings.get('check_bad_description', '0'))
checkBlacklistCompanies = parse_bool_setting(settings.get('check_blacklist_companies', '0'))
checkBlacklistTitles = parse_bool_setting(settings.get('check_blacklist_titles', '0'))
checkBlacklistDescription = parse_bool_setting(settings.get('check_blacklist_description', '0'))

# Keywords
goodTitles = parse_list_setting(settings.get('good_titles', ''))
goodDescriptions = parse_list_setting(settings.get('good_descriptions', ''))
badDescriptions = parse_list_setting(settings.get('bad_descriptions', ''))

# Search parameters
startAtIndex = int(settings.get('start_at_index', '0'))
jobsPerPage = int(settings.get('jobs_per_page', '25'))
botMinSpeed = float(settings.get('bot_min_speed', '0.3'))
botMaxSpeed = float(settings.get('bot_max_speed', '3'))

# Browser settings
headless = parse_bool_setting(settings.get('headless', 'False'))
chromeProfilePath = settings.get('chrome_profile_path', '')

# Blacklists
blacklistCompanies = parse_list_setting(settings.get('blacklist_companies', ''))
blackListTitles = parse_list_setting(settings.get('blacklist_titles', ''))
blackListDescription = parse_list_setting(settings.get('blacklist_description', ''))

# Webdriver Elements 
totalJobs = "//small"
offersPerPage = "//li[@data-occludable-job-id]"
outputFile = open("output.txt", "a+")
logging.basicConfig(level=logging.WARNING)

def jobsToPages(numOfJobs: str) -> int:
  number_of_pages = 1

  if (' ' in numOfJobs):
    spaceIndex = numOfJobs.index(' ')
    totalJobs = (numOfJobs[0:spaceIndex])
    totalJobs_int = int(totalJobs.replace(',', ''))
    number_of_pages = math.ceil(totalJobs_int/jobsPerPage)
  else:
      number_of_pages = int(numOfJobs)

  return number_of_pages

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

    if(headless):
        options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    if(len(chromeProfilePath)>0):
        initialPath = chromeProfilePath[0:chromeProfilePath.rfind("/")]
        profileDir = chromeProfilePath[chromeProfilePath.rfind("/")+1:]
        options.add_argument('--user-data-dir=' +initialPath)
        options.add_argument("--profile-directory=" +profileDir)
    else:
        options.add_argument("--incognito")
    return options


def check_whole_word(text, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.search(pattern, text.lower()) is not None

def scrape():
    global outputFile
    countJobs = 0
    url = linkedinJobLink
    
    driver.get(url)
    time.sleep(random.uniform(botMinSpeed, botMaxSpeed))

    totalJobs = driver.find_element(By.XPATH,'//small').text 
    totalPages = jobsToPages(totalJobs)
    prYellow("Parsing " +str(totalJobs)+ " jobs.")

    for page in range(totalPages):
        currentPageJobs = jobsPerPage * page
        url = url +"&start="+ str(currentPageJobs)
        driver.get(url)
        time.sleep(random.uniform(botMinSpeed, botMaxSpeed))

        offersPerPage = driver.find_elements(By.XPATH, '//li[@data-occludable-job-id]')
        offerIds = [(offer.get_attribute(
            "data-occludable-job-id").split(":")[-1]) for offer in offersPerPage]
        time.sleep(random.uniform(botMinSpeed, botMaxSpeed))

        for offer in offersPerPage:
            try:
                offerId = offer.get_attribute("data-occludable-job-id")
                offerIds.append(int(offerId.split(":")[-1]))
            except Exception as e: 
                continue            

        for jobID in offerIds:
            if(countJobs < startAtIndex):
                countJobs += 1
                prYellow("Skipping job at index " + str(countJobs))
                continue;

            offerPage = 'https://www.linkedin.com/jobs/view/' + str(jobID)
            driver.get(offerPage)
            time.sleep(random.uniform(botMinSpeed, botMaxSpeed))

            countJobs += 1
            prYellow("Checking job at index " + str(countJobs))

            jobProperties = getJobProperties(countJobs)
            jobDescription = getJobDescription()
            
            #first check if title and job description contain any of the goodTitles

            
            if checkTitle and not "blacklisted" in jobProperties.lower():                        
                foundGoodTitle=False
                for title in goodTitles:
                    if check_whole_word(jobProperties, title):
                        foundGoodTitle=True
                        break
                    
                if foundGoodTitle is False:
                    prYellow("No good title found in job title, skipping: " + str(offerPage))
                    continue
                        
            if checkDescription and not "blacklisted" in jobDescription.lower():
                foundGoodDesc=False
                for desc in goodDescriptions:
                    if check_whole_word(jobDescription, desc):
                        foundGoodDesc=True
                        break  
                        
                if foundGoodDesc is False:
                    prYellow("No good description found in job description, skipping: " + str(offerPage))
                    continue    
                    
            if checkBadDescription:
                foundBadDesc = False;         
                for title in badDescriptions:
                    if check_whole_word(jobDescription, title):
                        foundBadDesc = True
                        break
                        
                if foundBadDesc is True:
                    prYellow("Found bad title in jobDescription: " + title)
                    continue      
            
            if "blacklisted" in jobProperties.lower():
                prYellow("Blacklisted Job, skipped: " +str(offerPage) + " reason: " + jobProperties)
                continue
                
            if "blacklisted" in jobDescription.lower():
                prYellow("Blacklisted Job description, skipped!: " +str(offerPage) + " reason: " + jobProperties)
                continue
            
            jobAlreadySaved = False
            outputFile.seek(0, 0)
            fileContent = outputFile.read()                
            if str(jobID) in fileContent:
                jobAlreadySaved = True
                break

            if not jobAlreadySaved:
                outputFile.seek(0, 2)
                prGreen("Saved job to File: " + offerPage)
                outputFile.write(offerPage + "\n")
                outputFile.flush()
            else:
                prGreen("Job already saved: " + offerPage)

def getJobProperties(count):
    textToWrite = ""
    jobTitle = ""

    time.sleep(botMaxSpeed) # wait for page to load

    try:
        jobTitle = driver.find_element(By.XPATH, "//*[contains(@class, 'job-title')]").get_attribute("innerHTML").strip()
        if checkBlacklistTitles:
            res = [blItem for blItem in blackListTitles if check_whole_word(jobTitle, blItem)]
            if (len(res) > 0):
                jobTitle = "(blacklisted title: " + ' '.join(res) + ")"
    except Exception as e:
        prYellow("Warning in getting jobTitle: " + str(e)[0:50])
        jobTitle = ""

    try:
        jobCompanyName = driver.find_element(By.XPATH, "//*[contains(@class, 'company-name')]").get_attribute("innerHTML").strip()
        if checkBlacklistCompanies:
            res = [blItem for blItem in blacklistCompanies if check_whole_word(jobCompanyName, blItem)]
            if (len(res) > 0):
                jobCompanyName = "(blacklisted company: " + ' '.join(res) + ")"
    except Exception as e:
        print(e)
        prYellow("Warning in getting jobDetail: " + str(e)[0:100])
        jobCompanyName = ""

    if("blacklisted" in jobTitle):
        textToWrite = jobTitle
    elif("blacklisted" in jobCompanyName):
        textToWrite = jobCompanyName
    else:
        textToWrite = str(count) + " | " + jobTitle +" | " + jobCompanyName
    return textToWrite

def getJobDescription():
    description = " "
    try:
        description= driver.find_element(By.ID,"job-details").get_attribute("innerHTML").strip()
        if checkBlacklistDescription and len(blackListDescription) > 0:
            res = [blItem for blItem in blackListDescription if check_whole_word(description, blItem)]
            if (len(res)>0):
                description += "(blacklisted description: "+ ' '.join(res)+ ")"
                print("***** Blacklisted description: "+ ' '.join(res))
    except Exception as e:
        prYellow("Warning in getting job description: " +str(e)[0:50])
        description = ""
    return description

def show_login_popup():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("LinkedIn Login Required", 
                       "Please log in to LinkedIn in the browser window.\n\n"
                       "After successful login, press Enter in the terminal to continue.")
    root.destroy()

# main
start = time.time()
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()),options=chromeBrowserOptions())
driver.get("https://www.linkedin.com/login?trk=guest_homepage-basic_nav-header-signin")

# Check if credentials are provided
if linkedin_username and linkedin_password:
    # Automated login
    driver.find_element("id","username").send_keys(linkedin_username)
    driver.find_element("id","password").send_keys(linkedin_password)
    driver.find_element("xpath",'//button[@type="submit"]').click()
    time.sleep(botMaxSpeed) # wait for login to complete
else:
    # Manual login
    show_login_popup()
    input("Press Enter to continue...")


scrape()
end = time.time()
prYellow("---Took: " + str(round((time.time() - start)/60)) + " minute(s).")
outputFile.close()