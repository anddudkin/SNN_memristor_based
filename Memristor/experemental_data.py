import pickle

import torch
import matplotlib.pyplot as plt
import pandas

current, voltage = [], []
for i in range(20):
    excel_data = pandas.read_excel("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/эксп вах/3.xls",
                                   sheet_name='Append' + str(i))
    current.append(excel_data["AI"].tolist())
    voltage.append(excel_data["AV"].tolist())
for i, j in enumerate(voltage):
    # plt.semilogy(j, current[i])
    #plt.plot(j[:250], current[i][:250])
    pass

# plt.show()

for i, j in enumerate(voltage):
    # plt.semilogy(j, current[i])
    voltage[i] = voltage[i][:250]
    current[i] = current[i][:250]
print(voltage)
print(current)
print(len(voltage[2]))

for i in range(20):
    x = 0
    for j in range(250):

        if voltage[i][j-x] < 0:
            voltage[i].pop(j-x)
            current[i].pop(j-x)
            x += 1


for i, j in enumerate(voltage):
    # plt.semilogy(j, current[i])
    plt.plot(j[:70], current[i][:70])

plt.show()

for i, j in enumerate(voltage):
    # plt.semilogy(j, current[i])
    voltage[i] = voltage[i][:70]
    current[i] = current[i][:70]
voltage1=[]
current1=[]
for i, j in enumerate(voltage):
    if len(voltage[i])!=1:
        voltage1.append(voltage[i])
        current1.append(current[i])
voltage1=voltage1[1:]
current1=current1[1:]
with open("V_0_07.pkl", 'wb') as f:
    pickle.dump(voltage1, f)
with open("I_0_07.pkl", 'wb') as f:
    pickle.dump(current1, f)
for i, j in enumerate(voltage1):
    # plt.semilogy(j, current[i])
    plt.semilogy(j[:70], current1[i][:70])
plt.show()