from turtle import *

tracer(0)

left(90)
color('red')
for i in range(9):
    forward(22 * 10)
    right(90)
    forward(6 * 10)
    right(90)

penup()  # Поднять хвост

forward(1 * 10)
right(90)
forward(5 * 10)
left(90)

pendown()  # Опустить хвост

for i in range(9):
    forward(53 * 10)
    right(90)
    forward(75 * 10)
    right(90)

# Рисуем координатную сетку
penup()
for x in range(-100, 100):
    for y in range(-100, 100):
        goto(x * 10, y * 10)
        dot(3)

done()