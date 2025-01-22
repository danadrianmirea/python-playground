import requests

# Replace with your GitHub Personal Access Token
GITHUB_TOKEN = ""
GITHUB_API_URL = "https://api.github.com"

# Headers for authentication
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

def get_repositories():
    """Fetch all repositories (public and private) for the authenticated user."""
    repos = []
    page = 1
    while True:
        response = requests.get(
            f"{GITHUB_API_URL}/user/repos",
            headers=HEADERS,
            params={"per_page": 100, "page": page}
        )
        if response.status_code != 200:
            print(f"Failed to fetch repositories: {response.json()}")
            break

        data = response.json()
        if not data:
            break

        repos.extend(data)
        page += 1

    return repos

def delete_file_or_folder(owner, repo, path):
    """Delete a specific file or folder in a repository."""
    # Get the default branch (needed for deletion)
    response = requests.get(
        f"{GITHUB_API_URL}/repos/{owner}/{repo}",
        headers=HEADERS
    )
    if response.status_code != 200:
        print(f"Failed to fetch repository info for {repo}: {response.json()}")
        return

    default_branch = response.json().get("default_branch", "main")

    # Get the contents of the path
    response = requests.get(
        f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}",
        headers=HEADERS
    )
    if response.status_code == 404:
        print(f"No such path '{path}' in repository {repo}.")
        return
    elif response.status_code != 200:
        print(f"Failed to fetch path '{path}' in repository {repo}: {response.json()}")
        return

    contents = response.json()

    # If the path is a folder, delete each file individually
    if isinstance(contents, list):
        for item in contents:
            delete_file_or_folder(owner, repo, item["path"])
    else:
        # The path is a single file
        sha = contents["sha"]
        delete_file(owner, repo, path, sha, default_branch)

def delete_file(owner, repo, path, sha, branch):
    """Delete a single file from the repository."""
    response = requests.delete(
        f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}",
        headers=HEADERS,
        json={
            "message": f"Delete {path} via script",
            "branch": branch,
            "sha": sha
        }
    )
    if response.status_code == 200:
        print(f"Successfully deleted '{path}' in repository {repo}.")
    else:
        print(f"Failed to delete '{path}' in repository {repo}: {response.json()}")

def main():
    repos = get_repositories()
    print(f"Found {len(repos)} repositories.")

    for repo in repos:
        owner = repo["owner"]["login"]
        repo_name = repo["name"]

        print(f"Processing repository: {owner}/{repo_name}")
        delete_file_or_folder(owner, repo_name, ".github/workflows")
        delete_file_or_folder(owner, repo_name, ".github/actions")

if __name__ == "__main__":
    main()
