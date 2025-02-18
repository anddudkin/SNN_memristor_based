import numpy as np
c=np.linspace(0.00005,0.01,10)
print(c)

def find_nearest(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return val
f=find_nearest(c,8e-3)
print(f)
print(f in c)
