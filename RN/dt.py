fname = "digitos.txt"

with open(fname) as f:
    lines = f.readlines()
    f.close()

fp = open('./newData.txt', 'w')

for l in lines:
    l = l[1:]
    fp.write(l)

fp.close()
