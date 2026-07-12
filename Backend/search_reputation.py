import os

def search_files(directory, query):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') or file.endswith('.tsx'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if query in line:
                                print(f"{path}:{i+1}: {line.strip()}")
                except Exception as e:
                    pass

search_files('c:\\Users\\sahit\\OneDrive\\Desktop\\sentinel\\SentinelX', 'reputation_score')
