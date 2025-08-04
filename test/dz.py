
m = []
n = []
h = 0
f = open('26.txt')
c = int((f.readline().split())[1])
p = 500
for x in f.readlines():
    s = x.split()
    if s[2] == 'A':
        m.append([int(s[0]), int(s[1])])
    else:
        n.append([int(s[0]), int(s[1])])
m.sort()

n.sort()
k = 0
gg=0
for i in range(len(m)):
    c = c - m[i][0] * m[i][1]
    o = c
for j in range(len(n)):
    if o - n[j][0] * n[j][1] >= 0:
        o = o - n[j][0] * n[j][1]
        h = h + n[j][1]
        gg+=n[j][0] * n[j][1]
    else:
        for i in range(n[j][1]):
            if o - n[j][0] * i >= 0:
                o = o - n[j][0] * i
                h = h + i
            else:
                gg += n[j][0] * i
                print(gg)
                print(h)
                print(o)
                break
