import os
import shutil
import stat

def make_writable(func, path, exc_info):
    """Forcefully make files writable to delete them."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)

def delete_git_subfolders():
    """Remove all subfolders whose name ends with '.git'."""
    current_dir = os.getcwd()  # Get the current working directory
    for folder_name in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder_name)
        # Check if the folder is a directory and its name ends with '.git'
        if os.path.isdir(folder_path) and folder_name.endswith(".git"):
            try:
                shutil.rmtree(folder_path, onerror=make_writable)  # Remove the folder and its contents
                print(f"Deleted: {folder_path}")
            except Exception as e:
                print(f"Failed to delete {folder_path}: {e}")
        else:
            print(f"Skipping: {folder_path} (does not end with '.git')")

if __name__ == "__main__":
    delete_git_subfolders()
