import csv
import matplotlib.pyplot as plt
import math
# TYPE = 'gflops'
TYPE = 'time'
data = open('./omp4.csv')
data2 = open('./omp8.csv')
data3 = open('./omp12.csv')
csvIter = csv.reader(data, delimiter=',', quotechar='|')
csvIter2 = csv.reader(data2, delimiter=',', quotechar='|')
csvIter3 = csv.reader(data3, delimiter=',', quotechar='|')
dictionary = {}

for row in csvIter:
    if len(row):
        [x, time, name, cmps] = row
        if not name=='C library':
            name=name+' with 4 threads'
        if (not name in dictionary):
            dictionary[name] = {"time": [], "cmps": [], "x": []}
        dictionary[name]["time"].append(float(time))
        dictionary[name]["cmps"].append(float(cmps))
        dictionary[name]["x"].append(int(x))

for row in csvIter2:
    if len(row):
        [x, time, name, cmps] = row
        if name=='C library':
            continue
        name=name+' with 8 threads'
        if (not name in dictionary):
            dictionary[name] = {"time": [], "cmps": [], "x": []}

        dictionary[name]["time"].append(float(time))
        dictionary[name]["cmps"].append(float(cmps))
        dictionary[name]["x"].append(int(x))

for row in csvIter3:
    if len(row):
        [x, time, name, cmps] = row
        if name=='C library' or name=='merge_opti':
            continue
        name=name+' with 12 threads'
        if (not name in dictionary) or name=='C library':
            dictionary[name] = {"time": [], "cmps": [], "x": []}

        dictionary[name]["time"].append(float(time))
        dictionary[name]["cmps"].append(float(cmps))
        dictionary[name]["x"].append(int(x))
print(dictionary.keys())
fig, ax = plt.subplots()
for key in dictionary.keys():
    ax.plot([math.log(p,10) for p in dictionary[key]["x"]], dictionary[key]["cmps"], linewidth=2.0,marker="o",label=key)

ax.set(xlim=(1, 7), ylim=(0, 70000))
plt.xlabel('exponent of 10')
plt.ylabel('element(s) per second')
plt.legend()
# plt.xticks(list(range(0, max(x)+100, 200)),
#            [str(i) for i in range(0, max(x)+100, 200)])
plt.savefig('./Figures/threads.png')