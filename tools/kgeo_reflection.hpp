#pragma once
// Compile-time description of the .kgeo records (Sane C++ Reflection).
//
// One place names every field of GeoPixel and KGeoMeta. Exporters derive their
// layout from it (see kgeo_to_npy.cpp) instead of repeating it by hand, and the
// static_asserts below fail the build when the struct and this description drift:
// IsPacked holds only if the described members cover sizeof(T) exactly, so both a
// field added to the struct but not listed here and hidden padding are caught.
//
// Tags are part of the persisted description: never renumber or reuse them.

#include "../render_data.hpp"
#include "../third_party/sane_cpp/SaneCppReflection.h"

SC_REFLECT_STRUCT_VISIT(GeoPixel)
SC_REFLECT_STRUCT_FIELD(0, outcome)
SC_REFLECT_STRUCT_FIELD(1, _pad)
SC_REFLECT_STRUCT_FIELD(2, r)
SC_REFLECT_STRUCT_FIELD(3, redshift)
SC_REFLECT_STRUCT_FIELD(4, magnif)
SC_REFLECT_STRUCT_FIELD(5, phi_disk)
SC_REFLECT_STRUCT_FIELD(6, theta_esc)
SC_REFLECT_STRUCT_FIELD(7, phi_esc)
SC_REFLECT_STRUCT_FIELD(8, fp_dr_a)
SC_REFLECT_STRUCT_FIELD(9, fp_dphi_a)
SC_REFLECT_STRUCT_FIELD(10, fp_dr_b)
SC_REFLECT_STRUCT_FIELD(11, fp_dphi_b)
SC_REFLECT_STRUCT_FIELD(12, coverage)
SC_REFLECT_STRUCT_FIELD(13, sky_dth_a)
SC_REFLECT_STRUCT_FIELD(14, sky_dph_a)
SC_REFLECT_STRUCT_FIELD(15, sky_dth_b)
SC_REFLECT_STRUCT_FIELD(16, sky_dph_b)
SC_REFLECT_STRUCT_LEAVE()

SC_REFLECT_STRUCT_VISIT(KGeoMeta)
SC_REFLECT_STRUCT_FIELD(0, W)
SC_REFLECT_STRUCT_FIELD(1, H)
SC_REFLECT_STRUCT_FIELD(2, M_bh)
SC_REFLECT_STRUCT_FIELD(3, a_bh)
SC_REFLECT_STRUCT_FIELD(4, Q_bh)
SC_REFLECT_STRUCT_FIELD(5, Lam)
SC_REFLECT_STRUCT_FIELD(6, r_isco)
SC_REFLECT_STRUCT_FIELD(7, r_disk_in)
SC_REFLECT_STRUCT_FIELD(8, r_disk_out)
SC_REFLECT_STRUCT_FIELD(9, theta_obs)
SC_REFLECT_STRUCT_FIELD(10, phi_obs)
SC_REFLECT_STRUCT_FIELD(11, r_obs)
SC_REFLECT_STRUCT_LEAVE()

static_assert(SC::Reflection::ExtendedTypeInfo<GeoPixel>::IsPacked,
              "GeoPixel: the reflected fields no longer cover the struct (new field or padding)");
static_assert(SC::Reflection::ExtendedTypeInfo<KGeoMeta>::IsPacked,
              "KGeoMeta: the reflected fields no longer cover the struct (new field or padding)");
