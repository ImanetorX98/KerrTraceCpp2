#pragma once
// Small utilities: wall clock, child processes, clamp.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "common.hpp"

static double get_time() {
#if defined(_OPENMP) && !defined(USE_METAL) && !defined(USE_CUDA)
    return omp_get_wtime();
#else
    using C = std::chrono::steady_clock;
    static auto t0 = C::now();
    return std::chrono::duration<double>(C::now()-t0).count();
#endif
}

// Pass paths as argv entries: spaces and shell metacharacters are literal.
static int run_process(const std::vector<std::string>& args) {
    std::vector<char*> argv;
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
#ifdef _WIN32
    return int(_spawnvp(_P_WAIT, argv[0], argv.data()));
#else
    pid_t child;
    if (posix_spawnp(&child, argv[0], nullptr, nullptr, argv.data(), environ) != 0)
        return -1;
    int status=0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

template<class T> static T clamp(T v, T lo, T hi) {
    return v < lo ? lo : v > hi ? hi : v;
}
