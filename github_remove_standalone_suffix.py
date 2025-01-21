import requests

# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"

# Replace with your GitHub Personal Access Token
GITHUB_TOKEN = ""

# GitHub username
GITHUB_USER = "danadrianmirea"


def get_repos():
    """Fetch the list of repositories from the authenticated user's account."""
    url = f"{GITHUB_API_URL}/user/repos?visibility=all"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    # GitHub API supports pagination, so we loop through all pages if necessary
    repos = []
    while url:
        response = requests.get(url, headers=headers)
        print(f"GET {url} - Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error fetching repos: {response.json()}")
            return repos
        repos.extend(response.json())
        # Check if there's a next page
        url = response.links.get('next', {}).get('url', None)
    
    return repos

def rename_repo(repo_name, new_name):
    """Rename a repository on GitHub."""
    url = f"{GITHUB_API_URL}/repos/{GITHUB_USER}/{repo_name}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {"name": new_name}
    
    response = requests.patch(url, headers=headers, json=data)
    print(f"PATCH {url} - Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Successfully renamed {repo_name} to {new_name}")
    else:
        print(f"Failed to rename {repo_name} to {new_name}: {response.json()}")

def fix_trailing_dash(repo_name):
    """Remove trailing '-' from repository name if present."""
    if repo_name.endswith('-'):
        return repo_name[:-1]
    return repo_name

def main():
    repos = get_repos()
    if not repos:
        print("No repositories found or failed to fetch repositories.")
        return
    
    renamed_any = False
    for repo in repos:
        repo_name = repo["name"]
        print(f"Checking repository: {repo_name}")  # Debugging line
        
        # If the repo ends with '-standalone', rename it
        if repo_name.endswith("-standalone"):
            new_name = repo_name[:-10]  # Remove the '-standalone' suffix
            new_name = fix_trailing_dash(new_name)  # Remove any trailing dash if present
            rename_repo(repo_name, new_name)
            renamed_any = True
        
        # If the repo ends with '-', fix it by removing the trailing '-'
        elif repo_name.endswith('-'):
            new_name = fix_trailing_dash(repo_name)
            rename_repo(repo_name, new_name)
            renamed_any = True

    if not renamed_any:
        print("No repositories to rename (no repository with '-standalone' or trailing '-' found).")

if __name__ == "__main__":
    main()
