# Sane C++ Libraries (vendored)

`SaneCppReflection.h` is the single-file build of the Reflection library from
<https://github.com/Pagghiu/SaneCppLibraries>, MIT licensed (see `LICENSE.txt`).

- Version: `release/2026/08` (commit 61a46a65f), downloaded from the
  `release/2026/09` GitHub release assets on 2026-10-17.
- Used only by `tools/` (the `.kgeo` description and `kgeo_to_npy`); the renderer
  does not depend on it.
- To update: replace the file with a newer release asset and rebuild; the
  `static_assert`s in `tools/kgeo_reflection.hpp` must still pass.
