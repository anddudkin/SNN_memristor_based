import pickle

import torch
import matplotlib.pyplot as plt
import pandas
import pickle

# with open("G:/Другие компьютеры/Ноутбук/7сем/1_Магистратура/3 сем/Методы анализа микро- и наносистем/5_.txt") as f:
#     content = f.read()
# print(content)
excel_data = pandas.read_excel("G:/Другие компьютеры/Ноутбук/7сем/1_Магистратура/3 сем/Методы анализа микро- и наносистем/5.xls")
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
plt.legend()
plt.show()
#
# current, voltage, res = [], [],[]
# for i in range(15):
#     excel_data = pandas.read_excel("G:/Другие компьютеры/Ноутбук/7сем/1_Магистратура/3 сем/Методы анализа микро- и наносистем/5.xlsx",
#                                   )
#     current.append(excel_data["AI"].tolist())
#     voltage.append(excel_data["AV"].tolist())
#     res.append(excel_data["RES"].tolist())