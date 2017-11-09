fname = "../K_means/ex7data2.txt"

with open(fname) as f:
    lines = f.readlines()
    f.close()

fp = open('../K_means/newData.txt', 'w')

for l in lines:
    l = l[1:]
    fp.write(l)

fp.close()
