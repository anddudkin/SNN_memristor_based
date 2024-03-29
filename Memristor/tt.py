import badcrossbar
import numpy as np
in_n=130
out_n = 100
s=[]
s1=[]
for i in (5,10,15,20,30,40,50,60,70,80,90,100):
    applied_voltages = np.ones([in_n, 1])
    w = np.random.randint(20000,25000,(in_n,i))
    r_i = 1
    solution = badcrossbar.compute(applied_voltages, w, r_i)
    s.append(round(1-np.sum(solution.voltages.word_line)/i/in_n,3))
    s1.append(round(np.min(solution.voltages.word_line),3))
print(s)
print(s1)