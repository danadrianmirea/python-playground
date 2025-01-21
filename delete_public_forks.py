import requests

# Replace with your personal access token
GITHUB_TOKEN = ""
GITHUB_API_URL = "https://api.github.com"
USERNAME = "danadrianmirea"

def get_repositories():
    """Retrieve all repositories for the authenticated user."""
    url = f"{GITHUB_API_URL}/user/repos"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    params = {"per_page": 100, "visibility": "public"}
    repos = []
    
    while url:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch repositories: {response.json()}")
            return []
        repos.extend(response.json())
        url = response.links.get("next", {}).get("url")  # Handle pagination

    return repos

def delete_repository(repo_full_name):
    """Delete a repository given its full name."""
    url = f"{GITHUB_API_URL}/repos/{repo_full_name}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.delete(url, headers=headers)
    if response.status_code == 204:
        print(f"Successfully deleted repository: {repo_full_name}")
    else:
        print(f"Failed to delete repository {repo_full_name}: {response.json()}")

def main():
    repos = get_repositories()
    if not repos:
        print("No repositories found.")
        return

    for repo in repos:
        if repo["fork"]:  # Check if the repository is a fork
            print(f"Found fork: {repo['full_name']}")
            delete_repository(repo["full_name"])

if __name__ == "__main__":
    main()
