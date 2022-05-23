cmake_minimum_required(VERSION 3.16)
project(cuda_gemm CUDA C)

set(CMAKE_CUDA_STANDARD 14)

#[[setting your openBLAS generated path here]]
set(OpenBlasPath ~/github/OpenBLAS/build)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    MESSAGE("openmp found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

add_compile_options(-O3)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=compute_61")

#[[linking dynamic lib]]
link_libraries(m)
link_libraries(pthread)

#[[setting include path like '-I' in make]]
include_directories(${OpenBlasPath})
#[[setting link path like '-L' in make]]
link_directories(${OpenBlasPath}/lib)

add_library(ompGemm SHARED omp_gemm.c)
target_link_libraries(ompGemm libopenblas.a)


add_library(cudaGemm SHARED cuda_gemm.cu cuda_gemm_share.cu cublas_gemm.cu cuda_yours.cu)
set_target_properties(
        cudaGemm
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudaGemm ompGemm cublas)

add_executable(gemm_test main.c)
target_link_libraries(gemm_test ompGemm cudaGemm cublas)
#target_link_libraries(gemm_test cublas)



#[[
#]]
