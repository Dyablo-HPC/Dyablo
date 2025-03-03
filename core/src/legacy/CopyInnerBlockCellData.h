/**
 * \file CopyInnerBlockCellData.h
 * \author Pierre Kestener
 */
#ifndef COPY_INNER_BLOCK_CELL_DATA_FUNCTOR_H_
#define COPY_INNER_BLOCK_CELL_DATA_FUNCTOR_H_

#include "kokkos_shared.h"
#include "FieldManager.h"

// utils hydro
#include "utils_hydro.h"

// utils block
#include "legacy/utils_block.h"
#include "LegacyDataArray.h"

namespace dyablo { 


/*************************************************/
/*************************************************/
/*************************************************/
/**
 * For each oct of a group of octs, copy inner cell block data into a larger block data (ghost included).
 *
 * e.g. 3x3 block (no ghost) copied into a 5x5 block (with ghost, ghostwidth of 1)
 *            . . . . .
 * x x x      . x x x .
 * x x x  ==> . x x x . 
 * x x x      . x x x .
 *            . . . . .
 *
 * \note As always use the nested parallelism strategy:
 * - loop over octants              parallelized with Team policy,
 * - loop over cells inside a block paralellized with ThreadVectorRange policy.
 * 
 */
class CopyInnerBlockCellDataFunctor {

public:
  using index_t = int32_t;
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<index_t>>;
  using thread_t = team_policy_t::member_type;

  struct Params {
    int ndims;
    GravityType gravity_type;
  };

  /**
   * 
   *
   * \param[in] params
   * \param[in] U conservative variables - global block array data (no ghost)
   * \param[in] iGroup identify the group of octant we want to copy
   * \param[out] Ugroup conservative var of a group of octants (block data with ghosts)
   */
  CopyInnerBlockCellDataFunctor(Params    params,
                                id2index_t     fm,
                                blockSize_t    blockSizes,
                                uint32_t       ghostWidth,
                                uint32_t       nbOcts,
                                uint32_t       nbOctsPerGroup,
                                LegacyDataArray U,
                                DataArrayBlock Ugroup,
                                uint32_t       iGroup) :
    params(params),
    fm(fm), 
    blockSizes(blockSizes),
    ghostWidth(ghostWidth),
    nbOcts(nbOcts),
    nbOctsPerGroup(nbOctsPerGroup),
    U(U), 
    Ugroup(Ugroup),
    iGroup(iGroup),
    ndim(params.ndims)
  {
    copy_gravity = (params.gravity_type & GRAVITY_FIELD);
    ndim = params.ndims;
  };
  
  // static method which does it all: create and execute functor
  static void apply(Params    params,
		                id2index_t     fm,
                    blockSize_t    blockSizes,
                    uint32_t       ghostWidth,
                    uint32_t       nbOcts,
                    uint32_t       nbOctsPerGroup,
		                LegacyDataArray U,
                    DataArrayBlock Ugroup,
                    uint32_t       iGroup)
  {

    CopyInnerBlockCellDataFunctor functor(params, 
                                          fm,
                                          blockSizes,
                                          ghostWidth,
                                          nbOcts,
                                          nbOctsPerGroup,
                                          U,
                                          Ugroup,
                                          iGroup);
    
    uint32_t nbOctsInGroup = std::min( nbOctsPerGroup, nbOcts - iGroup*nbOctsPerGroup );
    team_policy_t policy (nbOctsInGroup,
                          Kokkos::AUTO() /* team size chosen by kokkos */);
    
    
    Kokkos::parallel_for("dyablo::CopyInnerBlockCellDataFunctor",
                         policy, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const
  {
    
    // iOct must span the range [iGroup*nbOctsPerGroup , (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup*nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_g = member.league_rank();
    
    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];
    
    //const int& bx_g = blockSizes_g[IX];
    //const int& by_g = blockSizes_g[IY];
    //const int& bz_g = blockSizes_g[IZ];
    
    uint32_t nbCells = ndim == 2 ? 
      bx * by : 
      bx * by * bz;
    //uint32_t nbCells_g = ndim == 2 ? bx_g*by_g : bx_g*by_g*bz_g;

      
    // perform vectorized loop inside a given block data
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, nbCells),
      [&](const index_t index) {

        // compute current cell coordinates inside block

        coord_t cell_coord = ndim == 2 ? 
          index_to_coord<2>(index, blockSizes) :
          index_to_coord<3>(index, blockSizes) ;

        // compute corresponding index in the block with ghost data
        uint32_t index_g = ndim == 2 ?
          coord_to_index_g<2>(cell_coord, blockSizes, ghostWidth) :
          coord_to_index_g<3>(cell_coord, blockSizes, ghostWidth) ;

        // get local conservative variable
        real_t d = U(index, fm[ID], iOct);
        Ugroup(index_g, fm[ID], iOct_g) = d;
        Ugroup(index_g, fm[IP], iOct_g) = U(index, fm[IP], iOct);
        Ugroup(index_g, fm[IU], iOct_g) = U(index, fm[IU], iOct);
        Ugroup(index_g, fm[IV], iOct_g) = U(index, fm[IV], iOct);

        if ( ndim == 3 )
          Ugroup(index_g, fm[IW], iOct_g) = U(index, fm[IW], iOct);

        if (copy_gravity) {
          Ugroup(index_g, fm[IGX], iOct_g) = U(index, fm[IGX], iOct);
          Ugroup(index_g, fm[IGY], iOct_g) = U(index, fm[IGY], iOct);
          if ( ndim == 3 )
            Ugroup(index_g, fm[IGZ], iOct_g) = U(index, fm[IGZ], iOct);
        }

      }); // end TeamVectorRange
      
    
  } // operator
  
  //! general parameters
  Params  params;
  
  //! field manager
  id2index_t   fm;

  //! block sizes without ghosts
  blockSize_t blockSizes;

  //! block sizes with    ghosts
  //blockSize_t blockSizes_g;

  //! ghost width
  uint32_t ghostWidth;

  //! total number of octants in current MPI process
  uint32_t nbOcts;

  //! number of octants per group
  uint32_t nbOctsPerGroup;

  //! heavy data - input - global array of block data (no ghosts)
  LegacyDataArray    U;

  //! heavy data - output - local group array of block data (with ghosts)
  DataArrayBlock    Ugroup;

  //! id of group of octants to be copied
  uint32_t iGroup;

  // should we copy gravity ?
  bool copy_gravity;

  // number of dimensions for gravity
  int ndim;

}; // CopyInnerBlockCellDataFunctor



} // namespace dyablo

#endif // COPY_INNER_BLOCK_CELL_DATA_FUNCTOR_H_
