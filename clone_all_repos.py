import os
import subprocess
import requests

# Your GitHub username and personal access token
GITHUB_USERNAME = "danadrianmirea"
GITHUB_TOKEN = ""

# GitHub API base URL
BASE_URL = "https://api.github.com"

# Headers for API authentication
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}"
}

def get_all_repos():
    """Fetch all repositories for the authenticated user."""
    repos = []
    page = 1
    while True:
        url = f"{BASE_URL}/user/repos?visibility=all&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            page_repos = response.json()
            if not page_repos:
                break
            repos.extend(page_repos)
            page += 1
        else:
            print(f"Failed to fetch repositories: {response.status_code} - {response.text}")
            break
    return repos

def clone_repo(repo_url, repo_name):
    """Clone a repository."""
    try:
        print(f"Cloning {repo_name}...")
        subprocess.run(["git", "clone", repo_url], check=True)
        print(f"Successfully cloned {repo_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {repo_name}: {e}")

def main():
    repos = get_all_repos()
    print(f"Found {len(repos)} repositories.")
    for repo in repos:
        repo_name = repo["name"]
        clone_url = repo["clone_url"]
        # Skip if the repo folder already exists
        if os.path.exists(repo_name):
            print(f"Skipping {repo_name}: folder already exists.")
        else:
            clone_repo(clone_url, repo_name)

if __name__ == "__main__":
    main()
