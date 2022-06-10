### 如何编译?

执行：`nvcc main.cu -Xcompiler -fopenmp -o a -lcublas -lcudart -gencode=arch=compute_61,code=compute_61`

### 如何启动master进程?

执行`deno run --allow-net --allow-read --allow-run deno.ts`

### 如何运行?

先运行master进程(如果没有deno 可以复制`/home/pp2019010070/node/bin/deno`到你的sbin里)(这个进程最好运行在a100n那台机器上,因为我在slaver里把master的ip写死了)

然后执行`./a`启动slaver进程(进程数量需要与deno.ts第四行里所写的maxDevice一致，默认为2node)
