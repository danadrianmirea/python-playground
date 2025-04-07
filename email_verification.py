import re
import dns.resolver
import smtplib

# Step 1: Validate email format
def is_valid_format(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email) is not None

# Step 2: Check if it's a supported email domain
def is_supported_domain(email):
    supported_domains = [
        # Microsoft domains
        'hotmail.com', 'outlook.com', 'live.com', 'msn.com',
        # Google domains
        'gmail.com',
        # Yahoo domains
        'yahoo.com', 'yahoo.co.uk', 'yahoo.co.jp', 'yahoo.fr', 'yahoo.de',
        # AOL domains
        'aol.com',
        # ProtonMail domains
        'protonmail.com', 'protonmail.ch',
        # Zoho domains
        'zoho.com',
        # Yandex domains
        'yandex.com', 'yandex.ru',
        # iCloud domains
        'icloud.com', 'me.com', 'mac.com'
    ]
    domain = email.split('@')[-1].lower()
    return domain in supported_domains

# Step 3: Check MX records
def has_mx_records(domain):
    try:
        answers = dns.resolver.resolve(domain, 'MX')
        return len(answers) > 0
    except:
        return False

# Step 4: Optional - Check via SMTP (limited reliability with Hotmail)
def smtp_check(email):
    domain = email.split('@')[1]
    try:
        mx_records = dns.resolver.resolve(domain, 'MX')
        mx_record = str(mx_records[0].exchange)

        # Connect to the mail server
        server = smtplib.SMTP()
        server.set_debuglevel(0)
        server.connect(mx_record)
        server.helo('example.com')
        server.mail('test@example.com')
        code, message = server.rcpt(email)
        server.quit()

        return code == 250 or code == 251  # 250 means OK, 251 is a forward
    except Exception as e:
        print(f"SMTP check failed: {e}")
        return False

# Main function
def verify_email(email, use_smtp=False):
    if not is_valid_format(email):
        return "Invalid email format"
    if not is_supported_domain(email):
        return "Not a supported email domain"
    if not has_mx_records(email.split('@')[1]):
        return "Domain does not accept emails"
    if use_smtp:
        if smtp_check(email):
            return "Email address appears valid (SMTP check)"
        else:
            return "SMTP check failed or email may not exist"
    return "Email is syntactically valid and domain is reachable"

# Example usage
if __name__ == "__main__":
    test_emails = [
        "test.user@hotmail.com",
        "invalid.email@",
        "someone@gmail.com",
        "test.person@outlook.com",
        "user@yahoo.com",
        "name@aol.com",
        "user@protonmail.com",
        "test@zoho.com",
        "user@yandex.com",
        "name@icloud.com"
    ]
    
    print("Testing multiple email addresses:")
    for email in test_emails:
        print(f"\nTesting: {email}")
        result = verify_email(email, use_smtp=False)  # Set to False to avoid SMTP timeouts
        print(f"Result: {result}")
