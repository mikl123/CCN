import subprocess
import os
import shutil

# List of script files to run
script_files = [
    'commands_same_2.txt',
    'commands_same_3.txt',
    'commands_same_4.txt',
    'commands_same_5.txt',
    'commands_same_6.txt'
]
for script in script_files:
    with open(script, 'r') as f:
        for line in f:
            cmd = line.strip()
            if cmd:
                subprocess.run(cmd, shell=True, check=True)
    # Rename dummyfolder to newfolder
    old_folder = 'hyp'
    new_folder = f'hyp_{script}'

    if os.path.exists(old_folder) and os.path.isdir(old_folder):
        shutil.move(old_folder, new_folder)
        print(f"Renamed {old_folder} to {new_folder}")
    else:
        print(f"Folder not found: {old_folder}")