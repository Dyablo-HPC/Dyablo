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
                               DataArrayBlock U_ghost,
                               DataArrayBlock Ugroup,
                               uint32_t iGroup) :
    pmesh(pmesh), params(params), 
    fm(fm), blockSizes(blockSizes), 
    ghostWidth(ghostWidth),
    nbOctsPerGroup(nbOctsPerGroup), 
    U(U), 
    U_ghost(U_ghost),
    Ugroup(Ugroup), 
    iGroup(iGroup)
  {
    bx_g = blockSizes[IX] + 2*ghostWidth;
    by_g = blockSizes[IY] + 2*ghostWidth;
    bz_g = blockSizes[IZ] + 2*ghostWidth;
  };

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap configMap, 
                    HydroParams params, 
                    id2index_t fm,
                    blockSize_t blockSizes,
                    uint32_t ghostWidth,
                    uint32_t nbOctsPerGroup,
                    DataArrayBlock U,
                    DataArrayBlock U_ghost,
                    DataArrayBlock Ugroup,
                    uint32_t iGroup)
  {

    CopyFaceBlockCellDataFunctor functor(pmesh, params, fm, 
                                         blockSizes, ghostWidth,
                                         nbOctsPerGroup, 
                                         U, U_ghost, 
                                         Ugroup, iGroup);

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
  void fill_ghost_face_2d_same_size(uint32_t iOct,
                                    uint32_t iOct_local,
                                    uint32_t iOct_neigh,
                                    bool     is_ghost,
                                    index_t  index,
                                    DIR_ID   dir,
                                    FACE_ID  face) const
  {
    
    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    // current octant and neighbor are at same level (= same size)
    if (index < ghostWidth*by and 
        dir == DIR_X          and 
        face == FACE_LEFT) {
      
      
      // border sizes for input  cell are : ghostWidth,by
      // border sizes for output cell are : ghostWidth,by+2*ghostWidth
      
      // compute cell coordinates inside border
      coord_t coord_border = index_to_coord(index, ghostWidth, by);
      
      // compute neighbor cell coordinates inside non-ghosted block
      coord_t coord_cur = {coord_border[IX],
                           coord_border[IY] + ghostWidth, 
                           0};
      
      // compute corresponding index in the ghosted block 
      uint32_t index_cur = coord_cur[IX] + (bx+2*ghostWidth)*coord_cur[IY];

      // shift border coords to access input (neighbor) cell data
      // on the right
      coord_border[IX] += (bx - ghostWidth);

      uint32_t index_border = coord_border[IX] + bx * coord_border[IY];

      if (is_ghost) {
        Ugroup(index_cur, fm[ID], iOct_local) = U_ghost(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U_ghost(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U_ghost(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U_ghost(index_border, fm[IV], iOct_neigh);
      } else {
        Ugroup(index_cur, fm[ID], iOct_local) = U(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U(index_border, fm[IV], iOct_neigh);
      }
    }
  
  } // fill_ghost_face_2d_same_size

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_face_2d(uint32_t iOct, 
                          uint32_t iOct_local, 
                          index_t  index_in, 
                          DIR_ID   dir, 
                          FACE_ID  face) const
  {

    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    // probe mesh neighbor information
    uint8_t iface = face + 2*dir;
    uint8_t codim = 1;

    // list of neighbors octant id, neighbor through a given face
    std::vector<uint32_t> neigh;
    std::vector<bool> isghost;

    pmesh->findNeighbours(iOct,iface,codim,neigh,isghost);

    /*
     * first deal with external border -- TODO
     */
    if (neigh.size() == 0) {
      // fill_ghost_face_2d_external_border(iOct, index_in, dir, face);
    } 
    
    /*
     * there one neighbor accross face , either same size or larger
     */
    else if (neigh.size() == 1) {

      // retrieve neighbor octant id
      uint32_t iOct_neigh = neigh[0];

      // check if neighbor is larger
      if ( pmesh->getLevel(iOct) > pmesh->getLevel(iOct_neigh) ) {

        // TODO

      } else {

        if (index_in==0)
          printf("[same size] iOct_global=%d iOct_local=%2d iOct_neigh=%2d \n",iOct, iOct_local, iOct_neigh);

        fill_ghost_face_2d_same_size(iOct, iOct_local, iOct_neigh, isghost[0], index_in, dir, face);

      } // end iOct and iOct_neigh have same size

    } // end neigh.size() == 1

    /*
     * there are 2 neighbors accross face, smaller than current octant
     */
    else if (neigh.size() == 2) {

      // TODO

    }

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

    // total number of octants
    uint32_t nbOcts = pmesh->getNumOctants();

    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    // maximun number of ghost cells to fill (per face)
    // this number will be used to set the number of working threads
    uint32_t bmax = bx < by ? by : bx;
    uint32_t nbCells = bmax*ghostWidth;

    while (iOct < iOctNextGroup and iOct < nbOcts) {

      // perform "vectorized" loop inside a given block data
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, nbCells),
          KOKKOS_LAMBDA(const index_t index) {

            // compute face X,left
            fill_ghost_face_2d(iOct, iOct_g, index, DIR_X, FACE_LEFT);

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

  //! ghost width
  uint32_t ghostWidth;

  //! block sizes with    ghosts
  uint32_t bx_g;
  uint32_t by_g;
  uint32_t bz_g;

  //! number of octants per group
  uint32_t nbOctsPerGroup;

  //! heavy data - input - global array of block data in octants own by current MPI process
  DataArrayBlock U;

  //! heavy data - input - global array of block data in MPI ghost octant
  DataArrayBlock U_ghost;

  //! heavy data - output - local group array of block data (with ghosts)
  DataArrayBlock Ugroup;

  //! id of group of octants to be copied
  uint32_t iGroup;

}; // CopyFaceBlockCellDataFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // COPY_FACE_BLOCK_CELL_DATA_FUNCTOR_H_
