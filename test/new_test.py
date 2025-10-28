import math

y = []

for i in range(100):
    with open("sin.txt","a") as f:
        f.write(f"{i};{round(math.sin(i),2)}\n")

