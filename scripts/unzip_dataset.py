import zipfile
import os
from pathlib import Path
import yaml

def extract_zip(zip_path, extract_to):
    print(f"[INFO] Extracting {zip_path} to {extract_to}...")
    if not os.path.exists(zip_path):
        print(f"[ERROR] Zip file {zip_path} does not exist!")
        return False
        
    Path(extract_to).mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.namelist()
        total = len(members)
        count = 0
        for member in members:
            # Skip macOS metadata files and folders
            if "__MACOSX" in member or ".DS_Store" in member.lower():
                continue
            
            # Target path
            target_path = os.path.join(extract_to, member)
            
            # If it's a directory, create it
            if member.endswith('/'):
                os.makedirs(target_path, exist_ok=True)
                continue
                
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Write file
            with z.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())
            
            count += 1
            if count % 500 == 0 or count == total:
                print(f"  Extracted {count}/{total} files...")
                
    print(f"[INFO] Successfully extracted {count} files from {zip_path}.")
    return True

def main():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "data" / "datasets" / "custom"
    
    # Extract images and labels
    images_zip = base_dir / "images.zip"
    labels_zip = base_dir / "labels.zip"
    
    success_imgs = extract_zip(str(images_zip), str(dataset_dir))
    success_lbls = extract_zip(str(labels_zip), str(dataset_dir))
    
    if not (success_imgs and success_lbls):
        print("[ERROR] Extraction failed.")
        return
        
    # Generate dataset.yaml
    classes = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches"
    ]
    dataset_yaml = {
        "path": dataset_dir.as_posix(),  # Use forward slashes
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes
    }
    
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)
        
    print(f"\n[INFO] Generated dataset.yaml at {yaml_path}")
    print(f"  Path: {dataset_yaml['path']}")
    print(f"  Classes: {dataset_yaml['names']}")

if __name__ == "__main__":
    main()
