# Post-build gate: assert a produced Mach-O binary really is arm64-only.
#
# The CMake-level guards decide what we *ask* the compiler for; this checks what
# actually came out, so a stray -arch, a toolchain wrapper or a cached flag
# cannot quietly reintroduce a Rosetta-dependent artefact.
#
# Invoked as: cmake -DKT_BIN=<path> -P cmake/verify_arch.cmake

if(NOT DEFINED KT_BIN)
    message(FATAL_ERROR "verify_arch.cmake: KT_BIN not set")
endif()

find_program(KT_LIPO lipo)
if(NOT KT_LIPO)
    # No lipo (non-Apple, or a stripped-down environment): nothing to verify.
    return()
endif()

execute_process(COMMAND "${KT_LIPO}" -archs "${KT_BIN}"
                OUTPUT_VARIABLE _archs
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _rc)

if(NOT _rc EQUAL 0)
    message(WARNING "verify_arch: could not read architectures of ${KT_BIN}")
    return()
endif()

if(NOT _archs MATCHES "arm64")
    message(FATAL_ERROR
        "${KT_BIN} was built for '${_archs}', not arm64.\n"
        "  That binary needs Rosetta and will not launch on macOS 28.")
endif()

if(_archs MATCHES "x86_64|i386")
    message(STATUS "verify_arch: ${KT_BIN} is universal (${_archs})")
endif()
