import os
import shutil
import stat

def is_git_repo(folder_path):
    """Check if the given folder is a Git repository."""
    return os.path.isdir(os.path.join(folder_path, ".git"))

def make_writable(func, path, exc_info):
    """Forcefully make files writable to delete them."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)

def delete_folder(folder_path):
    """Delete the specified folder."""
    try:
        shutil.rmtree(folder_path, onerror=make_writable)
        print(f"Deleted: {folder_path}")
    except Exception as e:
        print(f"Failed to delete {folder_path}: {e}")

def main():
    current_dir = os.getcwd()  # Get the current working directory
    for folder_name in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder_name)
        if os.path.isdir(folder_path) and is_git_repo(folder_path):
            print(f"Found Git repo: {folder_path}")
            delete_folder(folder_path)
        else:
            print(f"Skipping: {folder_path} (not a Git repo)")

if __name__ == "__main__":
    main()
