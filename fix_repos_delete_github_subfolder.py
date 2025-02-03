import requests

# Replace with your GitHub username and personal access token
GITHUB_USERNAME = ""
GITHUB_TOKEN = ""

# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"


# Get list of all repositories for the authenticated user
def get_repositories():
    repos = []
    page = 1
    while True:
        url = f"{GITHUB_API_URL}/user/repos?per_page=100&page={page}"
        response = requests.get(url, auth=(GITHUB_USERNAME, GITHUB_TOKEN))
        if response.status_code != 200:
            print(f"Error fetching repositories: {response.json()}")
            break

        data = response.json()
        if not data:
            break  # No more repositories

        repos.extend(data)
        page += 1

    return repos

# Recursively delete a folder and its contents
def delete_folder_contents(repo_name, folder_path):
    url = f"{GITHUB_API_URL}/repos/{GITHUB_USERNAME}/{repo_name}/contents/{folder_path}"
    response = requests.get(url, auth=(GITHUB_USERNAME, GITHUB_TOKEN))

    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            file_path = item["path"]
            file_url = f"{GITHUB_API_URL}/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}"

            if item["type"] == "file":
                delete_response = requests.delete(
                    file_url,
                    auth=(GITHUB_USERNAME, GITHUB_TOKEN),
                    json={"message": f"Removing {file_path}", "sha": item["sha"]}
                )
                if delete_response.status_code in [200, 204]:
                    print(f"Deleted {file_path} from {repo_name}")
                else:
                    print(f"Failed to delete {file_path} from {repo_name}: {delete_response.json()}")

            elif item["type"] == "dir":
                delete_folder_contents(repo_name, file_path)  # Recursively delete subfolders

    elif response.status_code == 404:
        print(f"No {folder_path} folder in {repo_name}, skipping.")
    else:
        print(f"Error checking {folder_path} in {repo_name}: {response.json()}")

# Main function
def main():
    repos = get_repositories()
    if not repos:
        print("No repositories found or failed to fetch repositories.")
        return

    print(f"Found {len(repos)} repositories.")
    for repo in repos:
        repo_name = repo["name"]
        print(f"Checking repository: {repo_name}")
        delete_folder_contents(repo_name, ".github")

if __name__ == "__main__":
    main()
