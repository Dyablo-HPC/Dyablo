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

// TODO find a way to avoid confusion between VarIndex position in View
using VarIndex = int;

} // namespace dyablo