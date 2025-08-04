for p in range(11,36):
    for x in range(0,p):
        for y in range(p):
            for z in range(p):
                if (int(str(z)+str(x), p) + int(str(x)+str(y), p)) == int(str(z)+str(y)+'a', p):
                    print(int(str(x)+str(y)+str(z), p), p)
