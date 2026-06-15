import subprocess
import time
import os

url = "https://files.pythonhosted.org/packages/a4/e4/9b378a5466ea0bed65e5beb8e09254973c580a6522810a38afbcc45e5105/onnxruntime_gpu-1.26.0-cp312-cp312-win_amd64.whl"
output_dir = "data/tmp"
os.makedirs(output_dir, exist_ok=True)
dest = os.path.join(output_dir, "onnxruntime_gpu-1.26.0-cp312-cp312-win_amd64.whl")

print(f"Downloading to {dest}...")
max_retries = 5
for attempt in range(max_retries):
    # Use curl with -L (follow redirect), -C - (resume), -o (output file)
    proc = subprocess.run(["curl", "-L", "-C", "-", "-o", dest, url])
    if proc.returncode == 0:
        print("Download completed successfully!")
        break
    print(f"Download interrupted (attempt {attempt + 1}/{max_retries}). Retrying in 5 seconds...")
    time.sleep(5)
else:
    print(f"Download failed after {max_retries} attempts.")
