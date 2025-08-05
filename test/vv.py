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
print(len(m))
print(len(n))
n.sort()
k = 0
for i in range(len(m)):
    c = c - m[i][0] * m[i][1]
o = c

for j in range(len(n)):
    if o - n[j][0] * n[j][1] >= 0:
        o = o - n[j][0] * n[j][1]
        h = h + n[j][1]
    else:
        for i in range(n[j][1]):
            if o - n[j][0] * i >= 0:
                o = o - n[j][0] * i
                h = h + i
            else:
                print(h)
                print(o)
                break
print(k)

breakpoint()
f = open("26.txt")
n, m = f.readline().split()
n = int(n);
m = int(m)
v = []
for i in range(n):
    a, b, c = f.readline().split()
    v.append((c, int(a), int(b)))
    v.sort(key=lambda x: (x[0], x[1]))
    be = 0
print(len(v))
for i in v:
    if m - i[1] < 0:
        break
    for z in range(i[2]):
        if m - i[1] >= 0:
            m = m - i[1]
            if i[0] == 'B':
                be += 1
print(be, m)
