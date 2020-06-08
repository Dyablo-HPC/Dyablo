/**
 * \file CopyFaceBlockCellDataHash.h
 * \author Pierre Kestener
 */
#ifndef COPY_FACE_BLOCK_CELL_DATA_HASH_FUNCTOR_H_
#define COPY_FACE_BLOCK_CELL_DATA_HASH_FUNCTOR_H_

#include "shared/FieldManager.h"
#include "shared/kokkos_shared.h"

// utils hydro
#include "shared/utils_hydro.h"

// utils block for :
// - enum NEIGH_LOC / NEIGH_SIZE
#include "muscl_block/utils_block.h"

#include "shared/AMRMetaData.h"

namespace dyablo
{
namespace muscl_block
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * For each oct of a group of octs, fill all faces of the block data associated to octant.
 *
 * e.g. right face ("o" symbol) of the 3x3 block (on the left) is copied into a 5x5 block (with ghost, left face)
 *  
 *            . . . . . 
 * x x o      o x x x . 
 * x x o  ==> o x x x . 
 * x x o      o x x x .
 *            . . . . .
 *
 * The main difficulty here is to deploy the entire combinatorics of
 * geometrical possibilities in terms of 
 * - size of neighbor octant, i.e.
 *   is neighbor octant small, same size or larger thant current octant,
 * - direction : face along X, Y or Z behave slightly differently, need to
 *   efficiently take symetries into account
 * - 2d/3d
 *
 * So we need to be careful, have good testing code.
 * See file test_CopyGhostBlockCellData.cpp
 *
 * \note As always use the nested parallelism strategy:
 * - loop over octants              parallelized with Team policy,
 * - loop over cells inside a block paralellized with ThreadVectorRange policy.
 *
 *
 * In reality, to simplify things, we assume block sizes are even integers (TBC, maybe no needed).
 *
 * \sa functor CopyInnerBlockCellDataFunctor
 *
 * This new version of CopyFaceBlockCellDataFunctor uses AMRMetaData
 * instead of bitpit mesh (not kokkos compatible).
 *
 * \tparam dim dimension is 2 or 3, necessary for AMRMetaData
 */
template<int dim>
class CopyFaceBlockCellDataHashFunctor
{

  public:
  static void apply(AMRMetaData<dim> mesh,
                    ConfigMap configMap, 
                    HydroParams params, 
                    id2index_t fm,
                    blockSize_t blockSizes,
                    uint32_t ghostWidth,
                    uint32_t nbOctsPerGroup,
                    DataArrayBlock U,
                    DataArrayBlock U_ghost,
                    DataArrayBlock Ugroup,
                    uint32_t iGroup,
                    FlagArrayBlock Interface_flags) {};


}; // CopyFaceBlockCellDataHashFunctor

} // namespace muscl_block

} // namespace dyablo

// 2d specialization
#include "CopyFaceBlockCellDataHash2d.h"

// 3d specialization
#include "CopyFaceBlockCellDataHash3d.h"

#endif // COPY_FACE_BLOCK_CELL_DATA_HASH_FUNCTOR_H_
