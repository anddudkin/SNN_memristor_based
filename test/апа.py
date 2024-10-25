import matplotlib.pyplot as plt
def cc(x, eps):
    g= []
    x1= ((x-1)/(x+1))*2
    n=3
    print(x1)
    while ((x-1)**n)/(3*(x+1)**n) > eps:

        x1+= ((x-1)**n)/(3*(x+1)**n)
        n+=2
        g.append(x1)
        print(x1)
    plt.plot(g)
    plt.show()
cc(10, 0.000000001)