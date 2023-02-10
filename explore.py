from sys import argv
in_file = argv[1]
file = open(in_file, 'r')
from json import load
config = load(file)

print(config)
