cmake_minimum_required(VERSION 3.18)

# ── Apple Silicon: build arm64, never Rosetta ─────────────────
# Rosetta 2 is supported through macOS 27 and removed in macOS 28, so an x86_64
# build of this project would simply stop launching. The project's own sources
# are architecture-neutral C++ -- what decides the output architecture is the
# toolchain, and that is what needs guarding.
#
# The trap: cmake itself may be an x86_64 binary running under translation (the
# Intel Homebrew one in /usr/local is a common case). CMake then believes it is
# on an x86_64 host and defaults the build to "-arch x86_64", silently producing
# a Rosetta-dependent binary. hw.optional.arm64 reports the truth even under
# translation, so ask the hardware rather than the process.
# Must run before project() so the compiler checks target the right architecture.
set(KT_HW_ARM64 OFF)
if(EXISTS "/usr/sbin/sysctl")
    execute_process(COMMAND /usr/sbin/sysctl -in hw.optional.arm64
                    OUTPUT_VARIABLE _kt_sysctl_arm64
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_kt_sysctl_arm64 STREQUAL "1")
        set(KT_HW_ARM64 ON)
    endif()
endif()

if(KT_HW_ARM64 AND NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING
        "macOS build architectures" FORCE)
endif()

project(KerrTracer VERSION 0.2.27 LANGUAGES CXX)

# Refuse to emit a binary that would need Rosetta. Cross-compiling for Intel
# stays possible, but only as a deliberate, explicit choice.
option(KERRTRACE_ALLOW_X86 "Permit an x86_64 build on Apple Silicon (needs Rosetta to run)" OFF)
if(KT_HW_ARM64 AND NOT KERRTRACE_ALLOW_X86)
    if(NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        message(FATAL_ERROR
            "Refusing to build for '${CMAKE_OSX_ARCHITECTURES}' on Apple Silicon: the\n"
            "  result would require Rosetta, which macOS 28 removes.\n"
            "  This usually means cmake itself is an x86_64 binary running under\n"
            "  translation. Check with:  lipo -archs \"$(command -v cmake)\"\n"
            "  and prefer a native one, e.g. /opt/homebrew/bin/cmake.\n"
            "  To override deliberately, configure with -DKERRTRACE_ALLOW_X86=ON.")
    endif()
    if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
        message(WARNING
            "cmake appears to be running under Rosetta (host reported as x86_64 on "
            "Apple Silicon hardware). The build has been pinned to arm64 and is "
            "correct, but cmake itself will stop working on macOS 28. "
            "Install a native cmake, e.g. 'brew install cmake' under /opt/homebrew.")
    endif()
endif()

if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    message(STATUS "macOS build architecture: ${CMAKE_OSX_ARCHITECTURES}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# ── CPU tuning flags ──────────────────────────────────────────
# -march=native is only valid when the compiler's target arch matches the host.
# Probe it instead of assuming, and fall back rather than failing the build.
include(CheckCXXCompilerFlag)
option(KERRTRACE_NATIVE_ARCH "Tune code generation for the building machine's CPU" ON)
set(KERRTRACE_ARCH_FLAGS "")
if(KERRTRACE_NATIVE_ARCH)
    check_cxx_compiler_flag("-march=native" KT_HAVE_MARCH_NATIVE)
    if(KT_HAVE_MARCH_NATIVE)
        set(KERRTRACE_ARCH_FLAGS -march=native)
    else()
        check_cxx_compiler_flag("-mcpu=native" KT_HAVE_MCPU_NATIVE)
        if(KT_HAVE_MCPU_NATIVE)
            set(KERRTRACE_ARCH_FLAGS -mcpu=native)
        else()
            message(STATUS "No usable native CPU tuning flag; building portable code")
        endif()
    endif()
endif()

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

# ── Post-build architecture gate ──────────────────────────────
# Verify the artefact that actually came out, not just the flags we asked for.
function(kt_verify_arch tgt)
    if(APPLE AND KT_HW_ARM64 AND NOT KERRTRACE_ALLOW_X86)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                    -DKT_BIN=$<TARGET_FILE:${tgt}>
                    -P ${CMAKE_SOURCE_DIR}/cmake/verify_arch.cmake
            COMMENT "Checking ${tgt} is arm64"
            VERBATIM)
    endif()
endfunction()

# ── Executable ────────────────────────────────────────────────
add_executable(kerr_tracer ${CPU_SOURCES})

target_include_directories(kerr_tracer PRIVATE ${CMAKE_SOURCE_DIR})

# Inject absolute output path so the binary finds it regardless of CWD
target_compile_definitions(kerr_tracer PRIVATE
    OUT_DIR="${CMAKE_SOURCE_DIR}/out")

target_compile_options(kerr_tracer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -O3 ${KERRTRACE_ARCH_FLAGS} -ffast-math
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

kt_verify_arch(kerr_tracer)

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
        $<$<COMPILE_LANGUAGE:CXX>:-O3 ${KERRTRACE_ARCH_FLAGS} -ffast-math -Wall -Wextra>)
    target_link_libraries(kerr_tracer_metal PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics")
    if(OpenMP_CXX_FOUND)
        target_link_libraries(kerr_tracer_metal PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(kerr_tracer_metal PRIVATE _OPENMP)
        message(STATUS "OpenMP enabled for kerr_tracer_metal fallback paths")
    endif()
    kt_verify_arch(kerr_tracer_metal)
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

    add_executable(kerrtrace_ray_bundle_regression
        tests/ray_bundle_regression.cpp)
    target_include_directories(kerrtrace_ray_bundle_regression
        PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(kerrtrace_ray_bundle_regression PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>)
    add_test(NAME kerrtrace.ray_bundle
        COMMAND kerrtrace_ray_bundle_regression $<TARGET_FILE:kerr_tracer>)
    # Three renders of a 320x180 frame: ~6 s when correct. The ceiling is here so
    # a non-terminating bundle fails the suite instead of hanging it.
    set_tests_properties(kerrtrace.ray_bundle PROPERTIES TIMEOUT 1200)

    add_executable(kerrtrace_chart_consistency_regression
        tests/chart_consistency_regression.cpp)
    target_include_directories(kerrtrace_chart_consistency_regression
        PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(kerrtrace_chart_consistency_regression PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>)
    add_test(NAME kerrtrace.chart_consistency
        COMMAND kerrtrace_chart_consistency_regression $<TARGET_FILE:kerr_tracer>)
    set_tests_properties(kerrtrace.chart_consistency PROPERTIES TIMEOUT 900)
endif()
