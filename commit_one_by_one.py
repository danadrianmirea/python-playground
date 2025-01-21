import os
import subprocess
import random

def is_git_repo(path):
    """Check if a directory is a Git repository."""
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def get_modified_and_untracked_files(path):
    """Get the list of modified and untracked files in the repository."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        modified_files = [line[3:] for line in lines if line.startswith(" M")]
        untracked_files_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        untracked_files = untracked_files_result.stdout.strip().split("\n")
        return modified_files + untracked_files
    except subprocess.CalledProcessError:
        return []

def commit_file(repo_path, file_path):
    """Commit a single file with a specific commit message and push it to the remote repository."""
    try:
        filename = os.path.basename(file_path)
        commit_message = f"updated {filename}"
        subprocess.run(["git", "add", file_path], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_path, check=True)
        subprocess.run(["git", "push"], cwd=repo_path, check=True)
        print(f"Committed {file_path} with message: '{commit_message}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to commit or push {file_path}: {e}")

def process_git_repo(repo_path, max_commits):
    """Check for local changes and commit them file by file, up to a random limit of commits."""
    files_to_commit = get_modified_and_untracked_files(repo_path)
    if not files_to_commit:
        print(f"No changes in repository: {repo_path}")
        return

    commit_count = 0
    for file in files_to_commit:
        if commit_count >= max_commits:
            print(f"Reached the maximum commit limit of {max_commits}.")
            break
        commit_file(repo_path, file)
        commit_count += 1

def main(start_path):
    """Main function to iterate over all subfolders and process Git repositories."""
    # Generate a random number of commits to make between 203 and 327
    max_commits = random.randint(203, 327)
    print(f"Limiting to {max_commits} commits in this run.")

    for root, dirs, _ in os.walk(start_path):
        # Check the topmost subfolder for being a Git repository
        if is_git_repo(root):
            print(f"Processing Git repository: {root}")
            process_git_repo(root, max_commits)
            # Skip descending into subfolders of this Git repo
            dirs[:] = []

if __name__ == "__main__":
    start_path = input("Enter the starting directory: ").strip()
    if os.path.isdir(start_path):
        main(start_path)
    else:
        print("Invalid directory path.")
