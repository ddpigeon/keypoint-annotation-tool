import os

# Define the directories
txt_dir = 'trial/labels'
jpg_dir = 'trial'

# Get lists of file names in each directory
txt_files = os.listdir(txt_dir)
jpg_files = os.listdir(jpg_dir)

# Iterate over the JPG files and check if each one has a corresponding TXT file
for jpg_file in jpg_files:
    if jpg_file.endswith('.jpg'):
        if jpg_file[:-4] not in [f.split('.')[0] for f in txt_files if f.endswith('.txt')]:
            # If there is no corresponding TXT file, delete the JPG file
            os.remove(os.path.join(jpg_dir, jpg_file))
            print(f'Removed: {jpg_file}')