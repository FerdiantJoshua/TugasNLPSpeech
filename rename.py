import os, re


PATH = './'
INCREASE_NUMBER = 95

regex = r'([a-z, A-Z]+)([0-9])(.wav)'
files = os.listdir(PATH)
result = []
for _file in files:
    name = re.match(regex, _file)
    if name:
        # result.append((_file, name.group(1), name.group(2), name.group(3)))
        os.rename(_file, '{}{}{}'.format(name.group(1), int(name.group(2)) + 95, name.group(3)))
# print(result)
# print(len(result))