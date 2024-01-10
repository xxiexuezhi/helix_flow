
import pickle
import sys

name = sys.argv[1]

with open(name,"rb") as f:
    d1= pickle.load(f)


print(d1)


for i in range(len(d1[-1][3])):
    print(d1[-1][3][i])

