import csv
import matplotlib.pyplot as plt
one = []
four = []
sixt = []
for row in open('./1.csv', newline='\r\n'):
    t, gf = row.split(',')
    t = float(t)
    gf = float(gf)*1000
    one.append(t)
for row in open('./4.csv', newline='\r\n'):
    t, gf = row.split(',')
    t = float(t)
    gf = float(gf)*1000
    four.append(t)
for row in open('./16.csv', newline='\r\n'):
    t, gf = row.split(',')
    t = float(t)
    gf = float(gf)*1000
    sixt.append(t)

x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
     1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
fig, ax = plt.subplots()
ax.plot(x, one, linewidth=1.8, label="1 process", marker="o")
ax.plot(x, four, linewidth=1.8, label="4 processes", marker=".")
ax.plot(x, sixt, linewidth=1.8, label="16 processes", marker="*")
plt.legend()
plt.xlabel('square matrix size')
plt.xticks(list(range(0, max(x)+100, 200)),
           [str(i) for i in range(0, max(x)+100, 200)])
plt.ylabel('GFLOPS')

ax.set(xlim=(0, 2100), ylim=(0, 25000*1.1))
plt.savefig('./3.png')
