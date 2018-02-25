import sys


lines = sys.stdin.readlines()

s = 0
for line in lines[:-1]:
    s += int(line.split()[0])

print(s/(len(lines)-1))
