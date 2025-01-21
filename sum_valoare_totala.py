from bs4 import BeautifulSoup

# Read the HTML content from a file
with open("data.html", "r", encoding="utf-8") as file:
    html_data = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_data, 'html.parser')

# Initialize a total sum variable
total_sum = 0.0

# Find all rows in the table
rows = soup.select("tbody tr")

# Iterate over rows and extract "Valoare totala" values
for row in rows:
    try:
        # Find all text in the first table cell
        first_cell = row.find("td")
        if first_cell:
            details = first_cell.get_text(strip=True)
            # Look for the line containing "Valoare totala"
            for line in details.split("\n"):
                if "Valoare totala:" in line:
                    # Extract the numeric value
                    value_str = line.split(":")[1].strip().split(" ")[0]
                    # Replace thousand separators (.) and decimal comma (,) for float conversion
                    value = float(value_str.replace(".", "").replace(",", "."))
                    total_sum += value
    except (AttributeError, IndexError, ValueError) as e:
        # Skip rows that don't match the expected format
        continue

# Print the total sum
print(f"Total value: {total_sum:.2f} RON")
