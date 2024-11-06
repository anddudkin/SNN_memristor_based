
import matplotlib.pyplot as plt
import numpy as np
import pandas

#excel_data = pandas.read_excel("G:/Другие компьютеры/Ноутбук/7сем/1_Магистратура/3 сем/Методы анализа микро- и наносистем/5.xls")
excel_data = pandas.read_excel("C:/Users/anddu/Desktop/7сем/1_Магистратура/3 сем/Методы анализа микро- и наносистем/5.xls")
print(excel_data)
excel_data=excel_data.drop('experimental', axis=1)
excel_data=excel_data.drop('smoothed', axis=1)
print(excel_data)
for i in excel_data:
    print(i)
    print(excel_data[i].tolist())
for i in excel_data:
    if i != "channel":
        plt.plot(excel_data["channel"].tolist(), excel_data[i].tolist(), label = i)
        #plt.semilogy(excel_data["channel"].tolist(), excel_data[i].tolist(), label=i) #log
plt.xlim(0, 2000)
plt.minorticks_on()
plt.xticks(np.linspace(0, 2000, 30))
plt.legend()
plt.show()
