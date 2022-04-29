import csv
import matplotlib.pyplot as plt

#TYPE = 'gflops'
TYPE = 'time'
mnpk = csv.reader(open('./omp/mnpk'), delimiter=',', quotechar='|')
mpnk = csv.reader(open('./omp/mpnk'), delimiter=',', quotechar='|')
pmnk = csv.reader(open('./omp/pmnk'), delimiter=',', quotechar='|')

x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
     1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
yt = []
ygflops = []


def gety(name):
    ygflops = []
    yt = []

    for row in name:
        if len(row):
            ygflops.append(float(row[2]))
            yt.append(float(row[1]))
    return [ygflops, yt]


fig, ax = plt.subplots()
# if TYPE == 'gflops':
#     y = ygflops
# else:
#     y = yt

ax.plot(x, gety(mnpk)[1], linewidth=2.0,label="inner",marker="o")
ax.plot(x, gety(mpnk)[1], linewidth=2.0,label="middle",marker=".")
ax.plot(x, gety(pmnk)[1], linewidth=2.0,label="outer",marker="*")
plt.legend()
plt.xlabel('square matrix size')
if TYPE == 'gflops':
    plt.ylabel('GFLOPS')
else:
    plt.ylabel('time (s)')
ax.set(xlim=(0, 2100), ylim=(0, 18*1.1))
plt.xticks(list(range(0, max(x)+100, 200)),
           [str(i) for i in range(0, max(x)+100, 200)])
plt.savefig('./figs/'+'3'+TYPE+'.png')
