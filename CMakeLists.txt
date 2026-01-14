cmake_minimum_required(VERSION 3.30)
project(OutAligner LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_CUDA_ARCHITECTURES 80)

find_package (OpenMP REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
#find_package (indicators REQUIRED)
#find_package (MPI REQUIRED)
#find_package (FMT REQUIRED)

include(FetchContent)

#set(JSON_BuildTests OFF CACHE INTERNAL "")
#FetchContent_Declare(json
#        GIT_REPOSITORY https://github.com/nlohmann/json.git
#        GIT_TAG v3.11.2
#)
#FetchContent_MakeAvailable(json)
FetchContent_Declare(argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

function(set_target_and_link TARGET_NAME TARGET_SOURCES)
    add_executable(${TARGET_NAME} ${TARGET_SOURCES})

    target_include_directories(${TARGET_NAME} PRIVATE include)
    include_directories(/home/yevhen/Documents/cutlass/include)
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    target_link_libraries(${TARGET_NAME} Eigen3::Eigen)
    target_link_libraries(${TARGET_NAME} OpenMP::OpenMP_CXX)
#    target_link_libraries(${TARGET_NAME} nlohmann_json::nlohmann_json)
    target_link_libraries(${TARGET_NAME} argparse)
#    target_link_libraries(${TARGET_NAME} indicators::indicators)
#    target_link_libraries(${TARGET_NAME} fmt::fmt-header-only)
#    target_link_libraries(${TARGET_NAME} MPI::MPI_CXX)

    target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
            $<$<COMPILE_LANGUAGE:CUDA>:-DCUTLASS_NVCC_ARCHS=80>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endfunction()


set(OUT_ALIGNER_BASE_SOURCES src/base_exec.cu)
set_target_and_link(OutAligner ${OUT_ALIGNER_BASE_SOURCES})

set(OUT_ALIGNER_BASE_SOURCES_LOCAL src/base_exec_local.cu)
set_target_and_link(OutAlignerLocal ${OUT_ALIGNER_BASE_SOURCES_LOCAL})


set(OUT_ALIGNER_VALIDATION_SOURCES src/validation.cu)
set_target_and_link(OutAlignerValidate ${OUT_ALIGNER_VALIDATION_SOURCES})

set(OUT_ALIGNER_VALIDATION_LOCAL_SOURCES src/validation_local.cu)
set_target_and_link(OutAlignerValidateLocal ${OUT_ALIGNER_VALIDATION_LOCAL_SOURCES})
