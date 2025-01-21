import requests

# Replace with your GitHub username and personal access token
username = "danadrianmirea"
token = ""

# GitHub API endpoint for user repositories
url = "https://api.github.com/user/repos"

# Parameters for pagination
params = {"per_page": 100}  # Fetch 100 repositories per page
all_repos = []  # Store all repositories

while url:
    # Make a GET request with authentication
    response = requests.get(url, auth=(username, token), params=params)

    if response.status_code == 200:
        # Add the repositories from the current page to the list
        all_repos.extend(response.json())

        # Check for the 'next' page in the response headers
        if "next" in response.links:
            url = response.links["next"]["url"]
        else:
            url = None  # No more pages
    else:
        print(f"Failed to fetch repositories: {response.status_code}")
        print(response.json())
        break

# Print all repository details
for repo in all_repos:
    print(f"{repo['html_url']}")

#print(f"Total Repositories: {len(all_repos)}")
