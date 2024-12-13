import time
start_time = time.time()
def ch(x):
    for i in range(len(x)-1):
        if x[i] == x[i+1]:
            return False
    return True
q = 0
sett = set()
for i1 in 'СВЕТЛАН':
    for i2 in 'СВЕТЛАН':
        for i3 in 'СВЕТЛАН':
            for i4 in 'СВЕТЛАН':
                for i5 in 'СВЕТЛАН':
                    for i6 in 'СВЕТЛАН':
                        for i7 in 'СВЕТЛАН':
                            for i8 in 'СВЕТЛАН':
                                a = i1+i2+i3+i4+i5+i6+i7+i8
                                #if a.count('А') == 2 and a.count('С') == 1 and a.count('В') == 1 and a.count('Е') ==1 and a.count('Т')==1 and a.count('Л') == 1 and a.count('Н') == 1:
                                if len(set(a)) == 7 and a.count("А") == 2:
                                    if i1 != i2 and i2 != i3 and i3 != i4 and i4 != i5 and i5 != i6 and i6 != i7 and i7 != i8 and (not(a in sett)):
                                    #if ch(a):
                                        sett.add(a)
                                        q += 1
print(q)
end_time = time.time()  # время окончания выполнения
execution_time = end_time - start_time  # вычисляем время выполнения

print(f"Время выполнения программы: {execution_time} секунд")