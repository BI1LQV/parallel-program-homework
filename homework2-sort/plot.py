import csv
import matplotlib.pyplot as plt
import math
FILE_NAME = 'ori.csv'
# TYPE = 'gflops'
TYPE = 'time'
data = open('./'+FILE_NAME)
csvIter = csv.reader(data, delimiter=',', quotechar='|')
dictionary = {}

for row in csvIter:
    if len(row):
        [x, time, name, cmps] = row
        if not name in dictionary:
            dictionary[name] = {"time": [], "cmps": [], "x": []}
        dictionary[name]["time"].append(float(time))
        dictionary[name]["cmps"].append(float(cmps))
        dictionary[name]["x"].append(int(x))

print(dictionary.keys())
fig, ax = plt.subplots()
for key in ['C library', 'quicksort_omp', 'mergesort_omp']:
    labeldic={
        'C library':'C library',
        'quicksort_omp':'quicksort',
        'mergesort_omp':'mergesort',
        'quicksort_opti':'quicksort_opti'
    }
    ax.plot([math.log(p,10) for p in dictionary[key]["x"]], dictionary[key]["cmps"], linewidth=2.0,marker="o",label=labeldic[key])

ax.set(xlim=(1, 7), ylim=(0, 65000))
plt.xlabel('exponent of 10')
plt.ylabel('element(s) per second')
plt.legend()
# plt.xticks(list(range(0, max(x)+100, 200)),
#            [str(i) for i in range(0, max(x)+100, 200)])
plt.savefig('./Figures/'+FILE_NAME+TYPE+'.png')