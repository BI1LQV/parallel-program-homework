import csv
import matplotlib.pyplot as plt

#TYPE = 'gflops'
TYPE = 'time'
mnpk = csv.reader(open('./1.csv'), delimiter=',', quotechar='|')
mpnk = csv.reader(open('./4.csv'), delimiter=',', quotechar='|')
pmnk = csv.reader(open('./16.csv'), delimiter=',', quotechar='|')

for row in open('./1.csv', newline='\r\n'):
    t, gf = row.split(',')
    t = float(t)
    gf = float(gf)
    print(gf)

exit(0)
x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
     1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
yt = []
ygflops = []


def gety(name):
    ygflops = []
    yt = []

    for row in name:
        if len(row):
            ygflops.append(float(row[1]*1000))
            yt.append(float(row[0]))
    return [ygflops, yt]


fig, ax = plt.subplots()
# if TYPE == 'gflops':
#     y = ygflops
# else:
#     y = yt

ax.plot(x, gety(mnpk)[1], linewidth=2.0, label="1", marker="o")
ax.plot(x, gety(mpnk)[1], linewidth=2.0, label="4", marker=".")
ax.plot(x, gety(pmnk)[1], linewidth=2.0, label="16", marker="*")
plt.legend()
plt.xlabel('square matrix size')
if TYPE == 'gflops':
    plt.ylabel('GFLOPS')
else:
    plt.ylabel('time (s)')
ax.set(xlim=(0, 2100), ylim=(0, 18*1.1))
plt.xticks(list(range(0, max(x)+100, 200)),
           [str(i) for i in range(0, max(x)+100, 200)])
plt.savefig('./3.png')
