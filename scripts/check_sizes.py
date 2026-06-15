import os
from pathlib import Path

def get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except PermissionError:
        pass
    return total

def main():
    root = Path("f:/steel-defect-detection")
    print("Steel Defect Detection Folder Size Analysis:")
    print("-" * 50)
    
    total_project_size = 0
    items = []
    
    for item in root.iterdir():
        if item.is_file():
            size = item.stat().st_size
            items.append((item.name, size, False))
            total_project_size += size
        elif item.is_dir():
            size = get_dir_size(item)
            items.append((item.name, size, True))
            total_project_size += size
            
    # Sort by size descending
    items.sort(key=lambda x: x[1], reverse=True)
    
    for name, size, is_dir in items:
        size_mb = size / (1024 * 1024)
        type_str = "[DIR]" if is_dir else "[FILE]"
        print(f"{type_str:<8} {name:<30} {size_mb:>10.2f} MB")
        
        # Subdirectories for data and frontend
        if is_dir and name in ["data", "frontend"]:
            sub_dir = root / name
            for entry in sub_dir.iterdir():
                if entry.is_dir():
                    sub_size = get_dir_size(entry)
                    print(f"  └─ [DIR]  {entry.name:<26} {sub_size / (1024 * 1024):>10.2f} MB")
                else:
                    sub_size = entry.stat().st_size
                    print(f"  └─ [FILE] {entry.name:<26} {sub_size / (1024 * 1024):>10.2f} MB")
        
    print("-" * 50)
    print(f"Total Project Size: {total_project_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
