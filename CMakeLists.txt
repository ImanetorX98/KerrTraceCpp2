cmake_minimum_required(VERSION 3.18)
project(KerrTracer VERSION 0.2.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# ── Backend selection ─────────────────────────────────────────
option(USE_METAL "Use Metal GPU backend (macOS only)"  OFF)
option(USE_CUDA  "Use CUDA GPU backend (Linux/Windows)" OFF)
option(CUDA_ENABLE_FAST_MATH "Enable --use_fast_math for CUDA kernels (faster, less precise)" OFF)

# ── Common CPU sources ────────────────────────────────────────
set(CPU_SOURCES main.cpp)

# ── OpenMP (CPU backend only) ─────────────────────────────────
find_package(OpenMP)

# ── Metal backend (macOS) ─────────────────────────────────────
if(USE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "USE_METAL requires macOS")
    endif()
    list(APPEND CPU_SOURCES gpu/metal/metal_renderer.mm)
    # Copy the .metal shader next to the binary so the renderer can find it
    configure_file(gpu/metal/tracer.metal
                   ${CMAKE_BINARY_DIR}/tracer.metal COPYONLY)
    message(STATUS "Metal GPU backend enabled")
endif()

# ── CUDA backend ──────────────────────────────────────────────
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Broad default to support most Windows/Linux NVIDIA cards out of the box.
        set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89" CACHE STRING
            "CUDA architectures to build for" FORCE)
    endif()
    list(APPEND CPU_SOURCES gpu/cuda/tracer.cu)
    # Separate compilation so CUDA sources can include our headers
    set_source_files_properties(gpu/cuda/tracer.cu
        PROPERTIES LANGUAGE CUDA)
    message(STATUS "CUDA backend enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# ── Executable ────────────────────────────────────────────────
add_executable(kerr_tracer ${CPU_SOURCES})

target_include_directories(kerr_tracer PRIVATE ${CMAKE_SOURCE_DIR})

# Inject absolute output path so the binary finds it regardless of CWD
target_compile_definitions(kerr_tracer PRIVATE
    OUT_DIR="${CMAKE_SOURCE_DIR}/out")

target_compile_options(kerr_tracer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -O3 -march=native -ffast-math
                                -Wall -Wextra>)

# Metal framework
if(USE_METAL)
    target_compile_definitions(kerr_tracer PRIVATE USE_METAL)
    target_link_libraries(kerr_tracer PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics")
endif()

# CUDA flags
if(USE_CUDA)
    target_compile_definitions(kerr_tracer PRIVATE USE_CUDA)
    target_compile_options(kerr_tracer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: -O3>)
    if(CUDA_ENABLE_FAST_MATH)
        target_compile_options(kerr_tracer PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)
    endif()
    set_target_properties(kerr_tracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()

# OpenMP (CPU path)
if(OpenMP_CXX_FOUND AND NOT USE_METAL AND NOT USE_CUDA)
    target_link_libraries(kerr_tracer PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(kerr_tracer PRIVATE _OPENMP)
    message(STATUS "OpenMP enabled (${OpenMP_CXX_VERSION})")
endif()

# ── Always build Metal binary on macOS (alongside CPU) ────────
if(APPLE AND NOT USE_METAL)
    add_executable(kerr_tracer_metal main.cpp gpu/metal/metal_renderer.mm)
    configure_file(gpu/metal/tracer.metal
                   ${CMAKE_BINARY_DIR}/tracer.metal COPYONLY)
    target_include_directories(kerr_tracer_metal PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_definitions(kerr_tracer_metal PRIVATE
        USE_METAL
        OUT_DIR="${CMAKE_SOURCE_DIR}/out")
    target_compile_options(kerr_tracer_metal PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -ffast-math -Wall -Wextra>)
    target_link_libraries(kerr_tracer_metal PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics")
    if(OpenMP_CXX_FOUND)
        target_link_libraries(kerr_tracer_metal PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(kerr_tracer_metal PRIVATE _OPENMP)
        message(STATUS "OpenMP enabled for kerr_tracer_metal fallback paths")
    endif()
    message(STATUS "Metal GPU binary: kerr_tracer_metal")
endif()

# ── Tests ───────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_executable(kerrtrace_core_tests tests/core_tests.cpp)
    target_include_directories(kerrtrace_core_tests PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(kerrtrace_core_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>)
    add_test(NAME kerrtrace.core COMMAND kerrtrace_core_tests)

    add_executable(kerrtrace_bump_detector tests/bump_detector.cpp)
    target_include_directories(kerrtrace_bump_detector PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(kerrtrace_bump_detector PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>)
    add_test(NAME kerrtrace.bump_detector COMMAND kerrtrace_bump_detector --self-test)

    add_executable(kerrtrace_spin_orientation_regression
        tests/spin_orientation_regression.cpp)
    target_include_directories(kerrtrace_spin_orientation_regression
        PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(kerrtrace_spin_orientation_regression PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>)
    add_test(NAME kerrtrace.spin_orientation
        COMMAND kerrtrace_spin_orientation_regression $<TARGET_FILE:kerr_tracer>)
endif()
