import numpy as np
import matplotlib.pyplot as plt

y_ax = np.linspace(0.1,4,200)
z_ax = np.linspace(0.1,12,200)

p =300
b = 3
m= b/2
n=b/2
x=0

result = []
for z in  z_ax:
    g = []
    for y in y_ax:
        g.append( p/(2*3.14) * (np.arctan(((x+m)*(y+n))/(z*np.sqrt((x+m)**2+(y+n)**2+z**2))) -
                      np.arctan(((x+m)*(y-n))/(z*np.sqrt((x+m)**2+(y-n)**2+z**2))) +
                      np.arctan(((x-m)*(y-n))/(z*np.sqrt((x-m)**2+(y-n)**2+z**2))) -
                      np.arctan(((x-m)*(y+n))/(z*np.sqrt((x-m)**2+(y+n)**2+z**2))) +
                      (z*(x+m)*(y+n)*((x+m)**2+(y+n)**2+2*z**2))/(((x+m)**2+z**2)*((y+n)**2+z**2)*(np.sqrt((x+m)**2)+(y+n)**2+z**2)) -
                      (z*(x+m)*(y-n)*((x+m)**2+(y-n)**2+2*z**2))/(((x+m)**2+z**2)*((y-n)**2+z**2)*(np.sqrt((x+m)**2)+(y-n)**2+z**2)) +
                      (z*(x-m)*(y-n)*((x-m)**2+(y-n)**2+2*z**2))/(((x-m)**2+z**2)*((y-n)**2+z**2)*(np.sqrt((x-m)**2)+(y-n)**2+z**2)) -
                      (z*(x-m)*(y+n)*((x-m)**2+(y+n)**2+2*z**2))/(((x-m)**2+z**2)*((y+n)**2+z**2)*(np.sqrt((x-m)**2)+(y+n)**2+z**2))
                      ))
    result.append(g)

print(result)

plt.imshow(result, vmin=0)
plt.show()
# locs, labels = plt.yticks()
# ax = plt.gca()
# ax.set_yticks(locs, np.linspace(1,12, 10))

# locs, labels = plt.yticks()
# print(locs)
# plt.yticks(np.linspace(0.1,12,len(locs)))


# for i in range(len(y_ax)):
#     plt.plot( y_ax, result[i])
# ax = plt.gca()
# ax.invert_xaxis()
# ax = plt.gca()
# ax.invert_yaxis()
# ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False, top=True, labeltop=True)
# plt.show()



# p =300
# b = 3
# m= b/2
# n=b/2
# y=2.5
# z=6
# x=0
# sigma = p/(2*3.14) * (np.arctan(((x+m)*(y+n))/(z*np.sqrt((x+m)**2+(y+n)**2+z**2))) -
#                       np.arctan(((x+m)*(y-n))/(z*np.sqrt((x+m)**2+(y-n)**2+z**2))) +
#                       np.arctan(((x-m)*(y-n))/(z*np.sqrt((x-m)**2+(y-n)**2+z**2))) -
#                       np.arctan(((x-m)*(y+n))/(z*np.sqrt((x-m)**2+(y+n)**2+z**2))) +
#                       (z*(x+m)*(y+n)*((x+m)**2+(y+n)**2+2*z**2))/(((x+m)**2+z**2)*((y+n)**2+z**2)*(np.sqrt((x+m)**2)+(y+n)**2+z**2)) -
#                       (z*(x+m)*(y-n)*((x+m)**2+(y-n)**2+2*z**2))/(((x+m)**2+z**2)*((y-n)**2+z**2)*(np.sqrt((x+m)**2)+(y-n)**2+z**2)) +
#                       (z*(x-m)*(y-n)*((x-m)**2+(y-n)**2+2*z**2))/(((x-m)**2+z**2)*((y-n)**2+z**2)*(np.sqrt((x-m)**2)+(y-n)**2+z**2)) -
#                       (z*(x-m)*(y+n)*((x-m)**2+(y+n)**2+2*z**2))/(((x-m)**2+z**2)*((y+n)**2+z**2)*(np.sqrt((x-m)**2)+(y+n)**2+z**2))
#                       )