#pragma once

#include <cmath>
#include <memory>

namespace dyablo {

// Conversion factors from unit to SI / code units
// Do not use in GPU kernels
namespace Units{

// SI units = 1
constexpr real_t meter = 1;
constexpr real_t second = 1;
constexpr real_t kilogram = 1;
constexpr real_t Kelvin = 1;

// Unit multiplicators
constexpr real_t Mega = 1e6;
constexpr real_t Kilo = 1e3;
constexpr real_t centi = 1e-2;
constexpr real_t milli = 1e-3;

// Units
constexpr real_t Newton = kilogram * meter / (second * second);
constexpr real_t Joule = Newton * meter;
constexpr real_t parsec = 3.085677e16 * meter;

// Constants
constexpr real_t KBOLTZ = 1.3806e-23 * Joule / Kelvin;
constexpr real_t PROTON_MASS = 1.67262158e-27 * kilogram;
constexpr real_t MHE_OVER_MH = 4.002;
constexpr real_t HELIUM_MASS = MHE_OVER_MH * PROTON_MASS;
constexpr real_t NEWTON_G = 6.67384e-11 * Newton * (meter*meter) / (kilogram*kilogram);
constexpr real_t SOLAR_MASS = 1.989e30 * kilogram;

}


} //namespace dyablo