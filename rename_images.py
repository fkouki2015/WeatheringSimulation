import os

folder = "/work/DDIPM/kfukushima/wsim/images_append"
start_number = 225

files = sorted(os.listdir(folder))

for i, filename in enumerate(files):
    src = os.path.join(folder, filename)
    if not os.path.isfile(src):
        continue
    ext = os.path.splitext(filename)[1].lower()
    new_name = f"image_{start_number + i}{ext}"
    dst = os.path.join(folder, new_name)
    print(f"{filename} -> {new_name}")
    os.rename(src, dst)

print(f"\nDone. {len(files)} files renamed.")
