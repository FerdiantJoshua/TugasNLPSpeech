import sys
import os

try:
    src_path = sys.argv[1]
    dest_path = sys.argv[2]
except IndexError:
    print('Usage:')
    print('\targ1 = source path')
    print('\targ2 = destination path')
    exit

for path, dirs, files in os.walk(src_path):
    for file in files:
        if '.wav' in file:
            os.rename(f'{path}/{file}', f'{dest_path}/{file[:-6].lower()}/{file}')