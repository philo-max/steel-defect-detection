import os
import sys
import time
import requests

URL = "https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.5.1%2Bcu121-cp312-cp312-win_amd64.whl"
FILENAME = "torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"

def download_file(url, dest):
    temp_dest = dest + ".tmp"
    headers = {}
    
    # Check if temp file exists to resume
    initial_pos = 0
    if os.path.exists(temp_dest):
        initial_pos = os.path.getsize(temp_dest)
        headers['Range'] = f'bytes={initial_pos}-'
        print(f"[INFO] Found temporary file. Resuming download from {initial_pos / 1024 / 1024:.2f} MB...")
    else:
        print(f"[INFO] Starting new download...")

    mode = 'ab' if initial_pos > 0 else 'wb'
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        # Check response code
        if response.status_code == 206:
            # Resuming successful
            total_size = int(response.headers.get('content-range').split('/')[-1])
        elif response.status_code == 200:
            # Range not supported or starting new
            total_size = int(response.headers.get('content-length', 0))
            if initial_pos > 0:
                print("[WARN] Server did not accept Range request, restarting download...")
                mode = 'wb'
                initial_pos = 0
        else:
            print(f"[ERROR] Server returned status code {response.status_code}")
            return False

        downloaded = initial_pos
        start_time = time.time()
        last_print = start_time

        with open(temp_dest, mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress every 5 seconds or when done
                    now = time.time()
                    if now - last_print > 5 or downloaded == total_size:
                        speed = (downloaded - initial_pos) / (now - start_time) if now - start_time > 0 else 0
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        eta = (total_size - downloaded) / speed if speed > 0 else 0
                        print(f"Progress: {downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB ({percent:.1f}%) | Speed: {speed / 1024:.1f} KB/s | ETA: {int(eta)}s")
                        last_print = now

        # Rename temp file to final destination on success
        if os.path.exists(dest):
            os.remove(dest)
        os.rename(temp_dest, dest)
        print(f"[SUCCESS] Download completed: {dest}")
        return True

    except Exception as e:
        print(f"[ERROR] Download exception: {e}")
        return False

def main():
    print(f"Downloading PyTorch GPU wheel...")
    print(f"Source: {URL}")
    print(f"Target: {FILENAME}")
    
    max_retries = 20
    retry_delay = 5
    
    for attempt in range(1, max_retries + 1):
        print(f"\n[Attempt {attempt}/{max_retries}] Starting download...")
        success = download_file(URL, FILENAME)
        if success:
            print("[INFO] PyTorch downloaded successfully.")
            sys.exit(0)
        else:
            print(f"[WARN] Attempt {attempt} failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            
    print("[ERROR] Max retries reached. Download failed.")
    sys.exit(1)

if __name__ == "__main__":
    main()
