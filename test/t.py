import numpy as np
c=np.linspace(0.00005,0.01,10)
print(c)
def bisection(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return array[0]
    elif (value > array[n-1]):
        return array[-1]
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return array[0]
    elif (value == array[n-1]):# and top
        return array[-1]
    else:
        return array[jl+1]
def find_nearest(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return val
f=find_nearest(c,8e-3)
print(f)
print(f in c)

f1=bisection(c,9e-3)
print(f1)
print(f1 in c)
