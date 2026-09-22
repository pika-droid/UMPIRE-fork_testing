import os
import zipfile
import time

zip_path = r'C:\Users\ashmi\Downloads\m3_generations.zip'
dest_dir = r'C:\Users\ashmi\OneDrive\Documents\umpire_testing\output\generations'

os.makedirs(dest_dir, exist_ok=True)
print(f"Extracting {zip_path} to {dest_dir}...")

t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as zf:
    members = zf.infolist()
    for member in members:
        zf.extract(member, dest_dir)
        print(f"Extracted {member.filename} ({member.file_size / (1024*1024):.2f} MB)")

print(f"Extraction complete in {time.time() - t0:.2f} seconds.")
