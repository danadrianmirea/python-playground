import requests

# Your GitHub username and personal access token
GITHUB_USERNAME = "danadrianmirea"
GITHUB_TOKEN = "your_personal_access_token"

# GitHub API base URL
BASE_URL = "https://api.github.com"

# Headers for API authentication
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}"
}

def make_repo_private(repo_name):
    """Make a repository private."""
    url = f"{BASE_URL}/repos/{GITHUB_USERNAME}/{repo_name}"
    response = requests.patch(url, headers=HEADERS, json={"private": True})
    if response.status_code == 200:
        print(f"Repository '{repo_name}' is now private.")
    else:
        print(f"Failed to make '{repo_name}' private: {response.status_code} - {response.text}")

def get_all_repos():
    """Get all repositories for the authenticated user."""
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

def main():
    repos = get_all_repos()
    print(f"Found {len(repos)} repositories.")
    for repo in repos:
        if not repo["private"]:
            make_repo_private(repo["name"])

if __name__ == "__main__":
    main()
