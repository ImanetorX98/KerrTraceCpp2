#pragma once
// Row-parallel loop shared by the phase-1 tracers.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "common.hpp"

// Calls row(py) once for every py in [0, H). Rows are independent, so they are
// handed out dynamically: rows near the black hole cost far more than sky rows.
template<class RowFn>
static void parallel_rows(int H, RowFn&& row) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 4)
    for (int py = 0; py < H; ++py) row(py);
#else
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned workers = std::max(1u, hw);
    if (workers == 1 || H < 4) {
        for (int py = 0; py < H; ++py) row(py);
        return;
    }
    std::atomic<int> next_row{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned t = 0; t < workers; ++t) {
        pool.emplace_back([&]() {
            while (true) {
                const int py = next_row.fetch_add(1);
                if (py >= H) break;
                row(py);
            }
        });
    }
    for (auto& th : pool) th.join();
#endif
}
