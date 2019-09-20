import os

BASE_PATH = 'raw_csv'
DIRECTORY_NAME = 'mfcc'

sound_type_count = {}

path, dirs, files = next(os.walk(f'{BASE_PATH}/{DIRECTORY_NAME}/'))
# print(len(files), path, dirs, files)

for file in files:
    sound_name = file.lower()[len(f'{DIRECTORY_NAME}-'):].replace('.csv', '')
    sound_name = ''.join(char for char in sound_name if not char.isdigit() and char not in ['(', ')', ' '])
    count = sound_type_count.get(sound_name)
    count = 1 if count == None else count + 1
    sound_type_count.update({sound_name:count})

total_count = 0
for sound_name, count in sound_type_count.items():
    total_count += count
    print(sound_name, count)
print(total_count)