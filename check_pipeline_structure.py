import os

print("[CHECK] Listing files in project root:")
project_root = os.path.dirname(os.path.abspath(__file__))
for root, dirs, files in os.walk(os.path.dirname(project_root)):
    level = root.replace(os.path.dirname(project_root), '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")
