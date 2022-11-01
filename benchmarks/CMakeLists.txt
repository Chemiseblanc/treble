FetchContent_Declare(
    GBench
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.7.0
)
FetchContent_MakeAvailable(GBench)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES "80")
    enable_language(CUDA)

    add_exe(benchmark_overhead
        SOURCES
            call_overheads.cu
        LINK
            treble::treble
            CUDA::cudart
            benchmark::benchmark
        PROPERTIES
            CUDA_ARCHITECTURES 80
    )
endif()

