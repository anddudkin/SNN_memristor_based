import pickle

import torch
import matplotlib.pyplot as plt
import pandas
import pickle

# with open("G:/Другие компьютеры/Ноутбук/7сем/1_Магистратура/3 сем/Методы анализа микро- и наносистем/5_.txt") as f:
#     content = f.read()
# print(content)
excel_data = pandas.read_excel("G:/Другие компьютеры/Ноутбук/7сем/1_Магистратура/3 сем/Методы анализа микро- и наносистем/5.xlsx",
                                  )
print(excel_data)
# import matplotlib.pyplot as plt
# import pandas
#
# current, voltage, res = [], [],[]
# for i in range(15):
#     excel_data = pandas.read_excel("G:/Другие компьютеры/Ноутбук/7сем/1_Магистратура/3 сем/Методы анализа микро- и наносистем/5.xlsx",
#                                   )
#     current.append(excel_data["AI"].tolist())
#     voltage.append(excel_data["AV"].tolist())
#     res.append(excel_data["RES"].tolist())