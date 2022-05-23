import matplotlib.pyplot as plt

txt=open('./res','r').read()
res=txt.split('\n')
gemm_openMP=[]
OpenBLAS=[]
cuda_global=[]
cuda_shared=[]
cublas=[]
l=[gemm_openMP,OpenBLAS,cuda_global,cuda_shared,cublas]
index=0
for line in res:
    if not line or 'Matrix' in line:
        continue
    l[index%5].append(line)
    index+=1


for g in l:
    for i in range(len(g)):
        g[i]=float(g[i].split(',')[0])
print(l)
x=range(100,2100,100)
fig, ax = plt.subplots()
ax.plot(x, l[0], linewidth=2.0,label="openMP",marker="o")
ax.plot(x, l[1], linewidth=2.0,label="OpenBLAS",marker=".")
ax.plot(x, l[2], linewidth=2.0,label="cuda_global",marker="*")
ax.plot(x, l[3], linewidth=2.0,label="cuda_shared",marker="+")
ax.plot(x, l[4], linewidth=2.0,label="cublas",marker="v")
plt.xlabel('square matrix size')
plt.ylabel('time (s)')
plt.legend()
ax.set(xlim=(100, 2100), ylim=(0, 1.77))
plt.xticks(list(range(200, 2100, 200)),
           [str(i) for i in range(200, 2100, 200)])
plt.savefig('./a2.png')