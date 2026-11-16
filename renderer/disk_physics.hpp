#pragma once
// Disk redshift and Doppler / magnification brightness factors.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "color_params.hpp"
#include "util.hpp"

static double disk_redshift(double r, double pt, double pphi, const KNdSMetric& g,
                             double observer_frequency) {
    return g.disk_frequency_shift(r, pt, pphi, observer_frequency);
}

static double doppler_luma_scale_inverted(double redshift, double exp) {
    // Physical g: >1 on approaching side (left for a>0), <1 on receding side.
    // Use g directly — no 2-g mirror needed.
    return std::pow(clamp(redshift, 0.01, 10.0), exp);
}

static inline double doppler_luma_scale(double redshift, const ColorParams& cp) {
    return cp.doppler_enabled ? doppler_luma_scale_inverted(redshift, cp.doppler_exp) : 1.0;
}

// Ray-bundle magnification as a luminance factor: OFF by default, because it is
// not physical for a resolved source.  Liouville's theorem makes I_nu/nu^3 constant
// along a null geodesic, so the observed surface brightness of the disk is exactly
// g^4 times the emitted one and does not depend on the bundle's footprint: lensing
// makes the disk cover more pixels, it does not make each pixel brighter.  The
// Jacobian |det J| is the footprint area, i.e. a filtering width (which is how DNGR
// uses it), not a flux correction.  Multiplying by 1/|det J| double-counted the
// lensing and produced the saturated blob at the photon ring.
//
// Kept as an explicit artistic option, reproducing the legacy factor verbatim:
// the raw Jacobian, clamped. No frame-wide normalisation is applied, so a pixel's
// colour depends only on that pixel's own geodesic.
static inline double magnif_luma_scale(double magnif, const ColorParams& cp) {
    if (!cp.bundle_magnif_luma) return 1.0;
    return clamp(1.0 / std::max(magnif, 1e-12), 0.05, std::max(0.05, cp.bundle_magnif_max));
}

// Continue tracing from an already-known state, but ignoring further disk hits.
// Used for alpha-compositing behind a partially transparent disk.
