import zipfile

def search_in_jar(jar_file, search_term):
    with zipfile.ZipFile(jar_file, 'r') as jar:
        for file in jar.namelist():
            with jar.open(file) as f:
                if search_term.encode() in f.read():
                    print(f"Found in: {file}")

search_in_jar("myfile.jar", "search-term")
