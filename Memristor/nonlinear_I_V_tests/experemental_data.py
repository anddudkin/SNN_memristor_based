import pickle

import torch
import matplotlib.pyplot as plt
import pandas

current, voltage, res = [], [],[]
for i in range(20):
    excel_data = pandas.read_excel("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/эксп вах/3.xls",
                                  sheet_name='Append' + str(i))
    current.append(excel_data["AI"].tolist())
    voltage.append(excel_data["AV"].tolist())
    res.append(excel_data["RES"].tolist())
print(len(voltage))
for i, j in enumerate(voltage):
    ii = [abs(ele) for ele in current[i]]
    plt.semilogy(j, ii, nonpositive='clip')
    plt.plot(j[:250], current[i][:250])
plt.rc('axes', labelsize=50)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Напряжение, В",fontsize=14)
plt.ylabel("Ток, А",fontsize=14)
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.tick_params(which='minor', direction="in")

plt.ylim(10**-6)
plt.show()

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
# with open("V_0_07.pkl", 'wb') as f:
#     pickle.dump(voltage1, f)
# with open("I_0_07.pkl", 'wb') as f:
#     pickle.dump(current1, f)
for i, j in enumerate(voltage1):
    # plt.semilogy(j, current[i])
    plt.semilogy(j[:70], current1[i][:70])
plt.ylim(10**-6)
plt.rc('axes', labelsize=50)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()