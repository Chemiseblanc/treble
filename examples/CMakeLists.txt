add_exe(basic_example
    SOURCES
        basic_example.cpp
    LINK
        treble::treble
)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES "80")
    enable_language(CUDA)

    add_exe(cuda_example
        SOURCES
            cuda_example.cu
        LINK
            treble::treble
            CUDA::cudart
        PROPERTIES
            CUDA_ARCHITECTURES 80
    )
endif()