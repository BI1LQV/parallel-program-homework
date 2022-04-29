import csv
import matplotlib.pyplot as plt

TYPE = 'gflops'
# TYPE = 'time'
data = open('./oppppp')
csvIter = csv.reader(data, delimiter=',', quotechar='|')
x = []

ygflops = []
y2 = []
switch=False
for row in csvIter:
    if len(row):
        if switch:
            x.append(int(row[0]))
            ygflops.append(float(row[2]))
        else:
            y2.append(float(row[2]))
        switch=not switch

fig, ax = plt.subplots()


ax.plot(x, ygflops, linewidth=2.0,marker="o",label="optimized_ref")
ax.plot(x, y2, linewidth=2.0,marker="o",label="origin_ref")
plt.xlabel('square matrix size')
plt.legend()
if TYPE == 'gflops':
    plt.ylabel('GFLOPS')
else:
    plt.ylabel('time (s)')
ax.set(xlim=(0, 2100), ylim=(0, max(ygflops)*1.1))
plt.xticks(list(range(0, max(x)+100, 200)),
           [str(i) for i in range(0, max(x)+100, 200)])
plt.savefig('./figs/sb.png')
