/**
 * \file CopyFaceBlockCellData.h
 * \author Pierre Kestener
 */
#ifndef COPY_FACE_BLOCK_CELL_DATA_FUNCTOR_H_
#define COPY_FACE_BLOCK_CELL_DATA_FUNCTOR_H_

#include "shared/FieldManager.h"
#include "shared/kokkos_shared.h"

// utils hydro
#include "shared/utils_hydro.h"

// utils block
#include "muscl_block/utils_block.h"

namespace dyablo {
namespace muscl_block {

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
 * \note As always use the nested parallelism strategy:
 * - loop over octants              parallelized with Team policy,
 * - loop over cells inside a block paralellized with ThreadVectorRange policy.
 *
 *
 * In reality, to simplify things, we assume block sizes are even integers.
 *
 */
class CopyFaceBlockCellDataFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using index_t = int32_t;
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<index_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  enum FACE_ID : uint8_t {
    FACE_LEFT=0,
    FACE_RIGHT=1
  };

  enum DIR_ID : uint8_t {
    DIR_X=0,
    DIR_Y=1,
    DIR_Z=2
  };

  /**
   *
   * \param[in] params
   * \param[in] U conservative variables - global block array data (no ghost)
   * \param[in] iGroup identify the group of octant we want to copy
   * \param[out] Ugroup conservative var of a group of octants (block data with
   *             ghosts)
   */
  CopyFaceBlockCellDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                               HydroParams params,
                               id2index_t fm,
                               blockSize_t blockSizes, 
                               uint32_t ghostWidth,
                               uint32_t nbOctsPerGroup,
                               DataArrayBlock U,
                               DataArrayBlock Ugroup,
                               uint32_t iGroup) :
    pmesh(pmesh), params(params), 
    fm(fm), blockSizes(blockSizes), ghostWidth(ghostWidth),
    nbOctsPerGroup(nbOctsPerGroup), U(U), Ugroup(Ugroup), iGroup(iGroup)
  {};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap configMap, 
                    HydroParams params, 
                    id2index_t fm,
                    blockSize_t blockSizes,
                    uint32_t ghostWidth,
                    uint32_t nbOctsPerGroup,
                    DataArrayBlock U,
                    DataArrayBlock Ugroup,
                    uint32_t iGroup)
  {

    CopyFaceBlockCellDataFunctor functor(pmesh, params, fm, blockSizes, ghostWidth,
                                         nbOctsPerGroup, U, Ugroup, iGroup);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("amr", "nbTeams", 16);
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_,
                         Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("dyablo::muscl_block::CopyFaceBlockCellDataFunctor",
                         policy, functor);
  }

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_face_2d(const uint32_t iOct, const index_t index, const DIR_ID dir, FACE_ID face) const 
  {

    

    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    if (face == FACE_LEFT) {
      // border sizes for input  cell are : ghostWidth,by
      // border sizes for output cell are : ghostWidth,by+2*ghostWidth


    }

    // compute current cell coordinates inside block
    coord_t cell_coord = index_to_coord<2>(index, blockSizes);
    
    // compute corresponding index in the block with ghost data
    uint32_t index_g = coord_to_index_g<2>(cell_coord, blockSizes, ghostWidth);
    
    // get local conservative variable
    // Ugroup(index_g, fm[ID], iOct_g) = U(index, fm[ID], iOct);
    // Ugroup(index_g, fm[IP], iOct_g) = U(index, fm[IP], iOct);
    // Ugroup(index_g, fm[IU], iOct_g) = U(index, fm[IU], iOct);
    // Ugroup(index_g, fm[IV], iOct_g) = U(index, fm[IV], iOct);
    
  } // fill_ghost_face_2d

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void functor2d(team_policy_t::member_type member) const {

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_g = member.league_rank();

    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    // maximun number of ghost cells to fill (per face)
    // this number will be used to set the number of working threads
    uint32_t bmax = bx < by ? by : bx;
    uint32_t nbCells = bmax*ghostWidth;

    while (iOct < iOctNextGroup) {

      // perform "vectorized" loop inside a given block data
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, nbCells),
          KOKKOS_LAMBDA(const index_t index) {

            // compute face X,left
            fill_ghost_face_2d(iOct, index, DIR_X, FACE_LEFT);

            // compute face X,right
            //fill_ghost_face_2d(iOct, index, DIR_X, FACE_RIGHT);

            // compute face Y,left
            //fill_ghost_face_2d(iOct, index, DIR_Y, FACE_LEFT);

            // compute face Y,right
            //fill_ghost_face_2d(iOct, index, DIR_Y, FACE_RIGHT);
            
          }); // end TeamVectorRange

      // increase current octant location both in U and Ugroup
      iOct += nbTeams;
      iOct_g += nbTeams;

    } // end while iOct inside current group of octants

  } // functor2d()

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const {

    if (params.dimType == TWO_D)
      functor2d(member);

    // if (params.dimType == THREE_D)
    //   functor3d(member);

  } // operator()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams params;

  //! field manager
  id2index_t fm;

  //! block sizes without ghosts
  blockSize_t blockSizes;

  //! block sizes with    ghosts
  // blockSize_t blockSizes_g;

  //! ghost width
  uint32_t ghostWidth;

  //! number of octants per group
  uint32_t nbOctsPerGroup;

  //! heavy data - input - global array of block data (no ghosts)
  DataArrayBlock U;

  //! heavy data - output - local group array of block data (with ghosts)
  DataArrayBlock Ugroup;

  //! id of group of octants to be copied
  uint32_t iGroup;

}; // CopyFaceBlockCellDataFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // COPY_FACE_BLOCK_CELL_DATA_FUNCTOR_H_
