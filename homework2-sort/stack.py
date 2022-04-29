import csv
import matplotlib.pyplot as plt
import math
file=open('./stack2','r')
ys=eval('['+file.read()+']')
fig, ax = plt.subplots()

ax.plot([i for i in range(len(ys))],ys,linewidth=1.0)

ax.set(xlim=(1, 1400000), ylim=(0, 50))
plt.xlabel('call count')
plt.ylabel('recursion depth')

plt.savefig('./Figures/stack2.png')