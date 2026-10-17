#pragma once
// Shared includes for the renderer modules.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "../camera.hpp"
#include "../geodesic.hpp"
#include "../knds_metric.hpp"
#include "../ray_bundle.hpp"
#include "../wormhole_metric.hpp"
#include "../render_data.hpp"


#include <atomic>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cerrno>
#ifdef _WIN32
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

#if defined(USE_METAL)
#  include "../gpu/metal/metal_renderer.hpp"
#elif defined(USE_CUDA)
#  include "../gpu/cuda/tracer.cuh"
#elif defined(_OPENMP)
#  include <omp.h>
#endif

#include <chrono>
