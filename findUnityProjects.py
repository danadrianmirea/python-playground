import os

def is_unity_project(folder):
    """Check if a folder is a Unity project by looking for key subfolders."""
    required_folders = {"Assets", "ProjectSettings", "assets", "projectSettings"}
    folder_contents = set(os.listdir(folder))

    return required_folders.issubset(folder_contents)

def find_unity_projects():
    """Iterate over subdirectories in the current directory and print Unity projects."""
    current_dir = os.getcwd()
    
    for subfolder in os.listdir(current_dir):
        full_path = os.path.join(current_dir, subfolder)
        if os.path.isdir(full_path) and is_unity_project(full_path):
            print(subfolder)

if __name__ == "__main__":
    find_unity_projects()
