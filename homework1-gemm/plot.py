import csv
import matplotlib.pyplot as plt
FILE_NAME = 'openblas'
TYPE = 'gflops'
# TYPE = 'time'
data = open('./'+FILE_NAME)
csvIter = csv.reader(data, delimiter=',', quotechar='|')
x = []
yt = []
ygflops = []
for row in csvIter:
    if len(row):
        x.append(int(row[0]))
        ygflops.append(float(row[2]))
        yt.append(float(row[1]))


fig, ax = plt.subplots()
if TYPE == 'gflops':
    y = ygflops
else:
    y = yt

ax.plot(x, y, linewidth=2.0)
plt.xlabel('square matrix size')
if TYPE == 'gflops':
    plt.ylabel('GFLOPS')
else:
    plt.ylabel('time (s)')
ax.set(xlim=(0, 2100), ylim=(0, max(y)*1.1))
plt.xticks(list(range(0, max(x)+100, 200)),
           [str(i) for i in range(0, max(x)+100, 200)])
plt.savefig('./figs/'+FILE_NAME+TYPE+'.png')
