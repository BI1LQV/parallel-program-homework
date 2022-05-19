from  lib import *
import matplotlib.pyplot as plt
import numpy as np
csv=open('./res.csv','r').read().split('\n')
OpenMP=[]
OpenBLAS=[]
CUDA_global_memory=[]
CUDA_shared_memory=[]
cuBLAS=[]
csrSpMM_serial=[]
csrSpMM_OpenMP=[]
csrSpMM_CUDA_scalar=[]
csrSpMM_CUDA_vector=[]
csrSpMM_cuSPARSE=[]

for line in csv:
    line=line.split(',')
    x=int(int(line[0])/100-1)
    y={'16':0,'32':1,'64':2}[line[1]]
    if y>=len(OpenMP):
        OpenMP.append([])
        OpenBLAS.append([])
        CUDA_global_memory.append([])
        CUDA_shared_memory.append([])
        cuBLAS.append([])
        csrSpMM_serial.append([])
        csrSpMM_OpenMP.append([])
        csrSpMM_CUDA_scalar.append([])
        csrSpMM_CUDA_vector.append([])
        csrSpMM_cuSPARSE.append([])

    OpenMP[y].append(float(line[3]))
    OpenBLAS[y].append(float(line[5]))
    CUDA_global_memory[y].append(float(line[7]))
    CUDA_shared_memory[y].append(float(line[9]))
    cuBLAS[y].append(float(line[11]))
    csrSpMM_serial[y].append(float(line[13]))
    csrSpMM_OpenMP[y].append(float(line[15]))
    csrSpMM_CUDA_scalar[y].append(float(line[17]))
    csrSpMM_CUDA_vector[y].append(float(line[19]))
    csrSpMM_cuSPARSE[y].append(float(line[21]))

y = ['16','32','64']
x = [i for i in list(range(100,2100,100))]

np.random.seed(196801)

fig, (ax, ax2,ax3, ax4,ax5) = plt.subplots(5, 1, figsize=(12, 10))
# im, _ = heatmap(np.array(OpenMP), y, x, ax=ax,
#                 cmap="Wistia", cbarlabel="OpenMP GFlops")
# annotate_heatmap(im, valfmt="{x:.4f}", size=7)



# im, _ = heatmap(np.array(OpenBLAS), y, x, ax=ax2,
#                 cmap="Wistia", cbarlabel="OpenBLAS GFlops")
# annotate_heatmap(im, valfmt="{x:.4f}", size=7)

# im, _ = heatmap(np.array(CUDA_global_memory), y, x, ax=ax3,
#                 cmap="Wistia", cbarlabel="CUDA_global GFlops")
# annotate_heatmap(im, valfmt="{x:.4f}", size=7)

# im, _ = heatmap(np.array(CUDA_shared_memory), y, x, ax=ax4,
#                 cmap="Wistia", cbarlabel="CUDA_shared GFlops")  
# annotate_heatmap(im, valfmt="{x:.4f}", size=7)

# im, _ = heatmap(np.array(cuBLAS), y, x, ax=ax5,
#                 cmap="Wistia", cbarlabel="cuBLAS GFlops")
# annotate_heatmap(im, valfmt="{x:.4f}", size=7)

im, _ = heatmap(np.array(csrSpMM_serial), y, x, ax=ax,
                cmap="Wistia", cbarlabel="csr_serial GFlops")
annotate_heatmap(im, valfmt="{x:.4f}", size=7)

im, _ = heatmap(np.array(csrSpMM_OpenMP), y, x, ax=ax2,
                cmap="Wistia", cbarlabel="csr_OpenMP GFlops")
annotate_heatmap(im, valfmt="{x:.4f}", size=7)

im, _ = heatmap(np.array(csrSpMM_CUDA_scalar), y, x, ax=ax3,
                cmap="Wistia", cbarlabel="csr_CUDA_scalar GFlops")
annotate_heatmap(im, valfmt="{x:.4f}", size=7)

im, _ = heatmap(np.array(csrSpMM_CUDA_vector), y, x, ax=ax4,
                cmap="Wistia", cbarlabel="csr_CUDA_vector GFlops")
annotate_heatmap(im, valfmt="{x:.4f}", size=7)

im, _ = heatmap(np.array(csrSpMM_cuSPARSE), y, x, ax=ax5,
                cmap="Wistia", cbarlabel="csr_cuSPARSE GFlops")
annotate_heatmap(im, valfmt="{x:.4f}", size=7)

plt.tight_layout()
plt.show()