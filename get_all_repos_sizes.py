import requests

# Replace with your personal GitHub token
token = ""
headers = {"Authorization": f"token {token}"}
url = "https://api.github.com/user/repos"

repos_data = []  # List to store repository data

while url:
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.json()}")
        break

    repos = response.json()  # Convert response to JSON
    for repo in repos:
        # Add the repository name and size (in KB) to the list
        repos_data.append({
            'name': repo.get('full_name', 'Unknown'),
            'size_kb': repo.get('size', 0)
        })

    # Check for pagination in the response headers
    url = response.links.get('next', {}).get('url') if 'next' in response.links else None

# Sort repositories by size in descending order
repos_data.sort(key=lambda x: x['size_kb'], reverse=True)

# Calculate total storage used
total_size_kb = sum(repo['size_kb'] for repo in repos_data)

# Output total storage and top 10 largest repositories
print(f"Total storage used: {total_size_kb / 1024:.2f} MB")
print("\nTop 10 largest repositories:")
for i, repo in enumerate(repos_data[:10], start=1):
    print(f"{i}. {repo['name']} - {repo['size_kb'] / 1024:.2f} MB")
