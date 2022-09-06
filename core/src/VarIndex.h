/**
 * VarIndex is user un FieldManager to translate Field names into indexes in multidimentionnal arrays
 * This file contains everything that needs to be modified to add a Field to a global array.
 * 
 * ** Read this before adding a VarIndex ** 
 * - Fields in VarIndex are meant to be used for GLOBAL fields, meaning fields that need to be conserved between iterations
 * - For local fields (internal to a kernel, used with a temporary array) only use local variables to store (unnamed) VarIndexes
 * - Use temporary arrays inside kernels as often as possible to avoid storing many variables in U 
 * 
 * To add an index:
 * (1) add an entry in the VarIndex enum before VARINDEX_COUNT
 * (2) add an entry in the var_names() vector
 * 
 **/

#pragma once

#include <vector>
#include <string>

namespace dyablo {

enum VarIndex : uint8_t {
  ID,           /// ID Density field index
  IP,           /// IP Pressure/Energy field index
  IE=IP,        /// IE Pressure/Energy field index
  IU,           /// X velocity / momentum index
  IV,           /// Y velocity / momentum index 
  IW,           /// Z velocity / momentum index 
  IBX,          /// X magnetic field index 
  IBY,          /// Y magnetic field index 
  IBZ,          /// Z magnetic field index
  IGPHI,        /// gravitational potential
  IGX,          /// X gravitational field index
  IGY,          /// Y gravitational field index
  IGZ,          /// Z gravitational field index
  /* (1) Add new index here */
  VARINDEX_COUNT/// invalid index, just counting number of fields
};

namespace{
[[maybe_unused]] const std::vector< std::pair< VarIndex, std::string > >& var_names()
{
  static std::vector< std::pair< VarIndex, std::string > > res{
    {ID, "rho"},
    {IU, "rho_vx"},
    {IV, "rho_vy"},
    {IW, "rho_vz"},
    {IE, "e_tot"},
    {IBX, "Bx"},
    {IBY, "By"},
    {IBZ, "Bz"},
    {IGPHI, "igphi"},
    {IGX, "igx"},
    {IGY, "igy"},
    {IGZ, "igz"}
    /* (2) Add a new (index, name) pair here*/
  };
  return res;
}
} // namespace

} // namespace dyablo