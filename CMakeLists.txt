cmake_minimum_required(VERSION 3.23)

project(sadcs VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMake's MSVC default adds /EHsc; strip it so our own /EHs-c- (no exceptions --
# sadcs never uses C++ exceptions) isn't overridden on the command line.
if(MSVC)
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

option(SADCS_AVX2 "Build with AVX2 (otherwise the SIMD helpers fall back to SSE2)" ON)
option(SADCS_CUDA "Build the GPU variant if a CUDA compiler is found" ON)

# The GPU variant is optional: without a CUDA compiler sadcs builds the CPU variant, and
# gpusa_stub.cpp reports every GPU call as unavailable. The CUDA 13.5 Visual Studio integration
# fails CMake's compiler check, so the GPU variant needs the Ninja-based cuda preset; the Visual
# Studio presets build the CPU variant.
if(SADCS_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(STATUS "sadcs: no CUDA compiler found, building the CPU variant only")
    endif()
endif()

# =============================================================================
# Source groups -- organized by concept (see README.md's directory layout).
# =============================================================================

# Shared by both variants: the difference cover tables (differencecover), SIMD
# byte compares (simd), the independent correctness checker (checker), and
# file I/O (fileio).
set(COMMON_SOURCES
    src/common/differencecover.h
    src/common/simd.h
    src/common/checker.h
    src/common/checker.cpp
    src/common/fileio.h
    src/common/fileio.cpp
)

# The CPU construction: the parallel building blocks (bucketsort, stringsort,
# parallelsort), the sample sort with its prefix doubling (samplesort,
# doubling), and the full construction (dcsort).
set(CPU_SOURCES
    src/cpu/bucketsort.h
    src/cpu/stringsort.h
    src/cpu/parallelsort.h
    src/cpu/doubling.h
    src/cpu/doubling.cpp
    src/cpu/samplesort.h
    src/cpu/inducedsort.h
    src/cpu/dcsort.h
    src/cpu/dcsort.cpp
)

# The sadcs.exe command line: commands (main) and the built-in adversarial
# test suite (selftest).
set(CLI_SOURCES
    src/cli/main.cpp
    src/cli/selftest.h
    src/cli/selftest.cpp
)

set(SADCS_SOURCES
    ${COMMON_SOURCES}
    ${CPU_SOURCES}
    ${CLI_SOURCES}
    src/cuda/gpusa.h
)

if(NOT CMAKE_CUDA_COMPILER)
    list(APPEND SADCS_SOURCES src/cuda/gpusa_stub.cpp)
endif()

add_executable(${PROJECT_NAME} ${SADCS_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE src/common src/cpu src/cuda src/cli)
source_group(TREE "${CMAKE_SOURCE_DIR}/src" FILES ${SADCS_SOURCES})

# The GPU variant: device code compiled by nvcc, with MSVC's cl.exe as host compiler on
# Windows. No OpenMP here. sm_120 (Blackwell) plus PTX for newer GPUs; CUB ships with the toolkit.
if(CMAKE_CUDA_COMPILER)
    add_library(sadcs_cuda STATIC src/cuda/gpusa.h src/cuda/gpusa.cu src/cuda/devicecommon.cuh
        src/cuda/samplerank.cuh src/cuda/classsort.cuh src/cuda/merge.cuh)
    set_target_properties(sadcs_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "120-real;120-virtual"
    )
    target_include_directories(sadcs_cuda PRIVATE src/common)
    if(MSVC)
        # CCCL 3 refuses cl.exe's traditional preprocessor.
        target_compile_options(sadcs_cuda PRIVATE -Xcompiler=/Zc:preprocessor,/utf-8)
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE sadcs_cuda)
endif()

if(MSVC)
    # Link-time code generation. The Visual Studio generator turns /GL into its
    # WholeProgramOptimization property, for clang-cl too; elsewhere clang-cl ignores /GL.
    if(CMAKE_GENERATOR MATCHES "Visual Studio" OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SADCS_LTCG ON)
    endif()

    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4
        /EHs-c-
        /utf-8
        $<$<CONFIG:Release>:/O2>
        $<$<AND:$<CONFIG:Release>,$<BOOL:${SADCS_LTCG}>>:/GL>
    )
    # _HAS_EXCEPTIONS=0: the MSVC STL's own switch matching /EHs-c- (otherwise C4530).
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _HAS_EXCEPTIONS=0
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<BOOL:${SADCS_LTCG}>>:/LTCG>
    )
    if(SADCS_AVX2)
        target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
    endif()

    # OpenMP. We need tasks (OpenMP 3.0+), so cl.exe's classic /openmp (2.0) won't do.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # clang-cl: LLVM's own libomp, found next to the compiler (bin/ -> lib/).
        get_filename_component(SADCS_LLVM_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(SADCS_LLVM_ROOT "${SADCS_LLVM_BIN}" DIRECTORY)
        find_library(SADCS_LIBOMP_LIB libomp PATHS "${SADCS_LLVM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
        find_file(SADCS_LIBOMP_DLL libomp.dll PATHS "${SADCS_LLVM_BIN}" NO_DEFAULT_PATH REQUIRED)
        # Not /openmp: the VS generator turns that into the OpenMPSupport property, which the
        # ClangCL toolset silently drops (the pragmas then compile to nothing). /clang: passes
        # the driver flag through untouched.
        target_compile_options(${PROJECT_NAME} PRIVATE /clang:-fopenmp)
        target_link_libraries(${PROJECT_NAME} PRIVATE "${SADCS_LIBOMP_LIB}")
        set(SADCS_OMP_DLLS "${SADCS_LIBOMP_DLL}")
    else()
        # cl.exe: /openmp:llvm links libomp140; the DLL lives in VS's redist tree, not on PATH.
        target_compile_options(${PROJECT_NAME} PRIVATE /openmp:llvm)
        get_filename_component(SADCS_VC_DIR "${CMAKE_CXX_COMPILER}/../../../../../../.." ABSOLUTE)
        file(GLOB SADCS_OMP_DLLS
            "${SADCS_VC_DIR}/Redist/MSVC/*/debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/libomp140*.x86_64.dll")
        if(NOT SADCS_OMP_DLLS)
            message(WARNING "libomp140.x86_64.dll not found under ${SADCS_VC_DIR}/Redist; sadcs.exe needs it on PATH")
        endif()
    endif()

    # The OpenMP runtime DLL has to sit next to sadcs.exe.
    foreach(dll ${SADCS_OMP_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            VERBATIM
        )
    endforeach()
    install(FILES ${SADCS_OMP_DLLS} DESTINATION bin)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -fno-exceptions $<$<CONFIG:Release>:-O2>)
    if(SADCS_AVX2)
        target_compile_options(${PROJECT_NAME} PRIVATE -mavx2 -mbmi)
    endif()
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(FILES LICENSE THIRD-PARTY-NOTICES.txt DESTINATION .)
