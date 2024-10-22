/**
 * \file ConvertToPrimitivesHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef CONVERT_TO_PRIMITIVES_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
#define CONVERT_TO_PRIMITIVES_HYDRO_MUSCL_BLOCK_FUNCTOR_H_

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "states/State_hydro.h"

#include "amr/AMRmesh.h"

// utils hydro
#include "utils_hydro.h"

// utils block
#include "legacy/utils_block.h"

namespace dyablo { 


/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Convert conservative variables to primitive variables functor.
 *
 */
class ConvertToPrimitivesHydroFunctor {

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  struct Params{
    int ndim;
    real_t gamma0, smallr, smallp;
  };

  /**
   * Convert conservative variables to primitive ones using equation of state.
   *
   * \param[in] params
   * \param[in] Udata conservative variables
   * \param[out] Qdata primitive variables
   */
  ConvertToPrimitivesHydroFunctor(Params    params,
				                          id2index_t     fm,
                                  blockSize_t    blockSizes,
                                  uint32_t       ghostWidth,
                                  uint32_t       nbOcts,
                                  uint32_t       nbOctsPerGroup,
                                  uint32_t       iGroup,
                                  DataArrayBlock Ugroup,
                                  DataArrayBlock Qgroup) :
    params(params), fm(fm), blockSizes(blockSizes), ghostWidth(ghostWidth),
    nbOcts(nbOcts), nbOctsPerGroup(nbOctsPerGroup),
    iGroup(iGroup), Ugroup(Ugroup), Qgroup(Qgroup)
  {

    const int bx_g = blockSizes[IX] + 2 * ghostWidth;
    const int by_g = blockSizes[IY] + 2 * ghostWidth;
    const int bz_g = blockSizes[IZ] + 2 * ghostWidth;
    
    nbCellsPerBlock = params.ndim==2 ? 
      bx_g * by_g :
      bx_g * by_g * bz_g;

  };
  
  // static method which does it all: create and execute functor
  static void apply(Params         params,
                    id2index_t     fm,
                    blockSize_t    blockSizes,
                    uint32_t       ghostWidth,
                    uint32_t       nbOcts,
                    uint32_t       nbOctsPerGroup,
                    uint32_t       iGroup,
                    DataArrayBlock Ugroup,
                    DataArrayBlock Qgroup)
  {

    ConvertToPrimitivesHydroFunctor functor(params, fm, blockSizes, ghostWidth, 
                                            nbOcts, nbOctsPerGroup, iGroup,
                                            Ugroup, Qgroup);
    
    uint32_t nbOctsInGroup = std::min( nbOctsPerGroup, nbOcts - iGroup*nbOctsPerGroup );
    team_policy_t policy (nbOctsInGroup,
                          Kokkos::AUTO() /* team size chosen by kokkos */);
    
    
    Kokkos::parallel_for("dyablo::ConvertToPrimitivesHydroFunctor",
                         policy, functor);
  }

  // ========================================================================
  // ========================================================================
  template <
    int ndim,
    typename PrimState,
    typename ConsState >
  KOKKOS_INLINE_FUNCTION
  void cons2prim(const int32_t index,
                 const uint32_t iOct_local) const
  {
    ConsState uLoc; // conservative variables in current cell
    PrimState qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    
    // get local conservative variable
    uLoc.rho   = Ugroup(index, fm[ID], iOct_local);
    uLoc.e_tot = Ugroup(index, fm[IP], iOct_local);
    uLoc.rho_u = Ugroup(index, fm[IU], iOct_local);
    uLoc.rho_v = Ugroup(index, fm[IV], iOct_local);
    if (ndim == 3)
      uLoc.rho_w = Ugroup(index, fm[IW], iOct_local);
    
    // get primitive variables in current cell
    computePrimitives<PrimState, ConsState>(uLoc, &c, qLoc, params.gamma0, params.smallr, params.smallp);
    
    // copy q state in q global
    Qgroup(index, fm[ID], iOct_local) = qLoc.rho;
    Qgroup(index, fm[IP], iOct_local) = qLoc.p;
    Qgroup(index, fm[IU], iOct_local) = qLoc.u;
    Qgroup(index, fm[IV], iOct_local) = qLoc.v;
    if (ndim == 3)
      Qgroup(index, fm[IW], iOct_local) = qLoc.w;
    
  } // cons2prim_3d

  // ========================================================================
  // ========================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const
  {

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    // uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, nbCellsPerBlock),
      [&](const int32_t index) {

        if (params.ndim==2)
          cons2prim<2, PrimHydroState, ConsHydroState>(index, iOct_local);
        else 
          cons2prim<3, PrimHydroState, ConsHydroState>(index, iOct_local);

      }); // end TeamVectorRange
  } // operator()

  //! general parameters
  Params  params;
  
  //! field manager
  id2index_t   fm;

  //! block sizes (no ghost)
  blockSize_t blockSizes;

  //! ghost width
  uint32_t ghostWidth;

  //! total number of octants in current MPI process
  uint32_t nbOcts;

  //! number of octant per group
  uint32_t nbOctsPerGroup;

  //! number of cells per block
  uint32_t nbCellsPerBlock;

  //! integer which identifies a group of octants
  uint32_t iGroup;

  //! heavy data - conservative variables
  DataArrayBlock    Ugroup;

  //! heavy data - primitive variables
  DataArrayBlock    Qgroup;

}; // ConvertToPrimitivesHydroFunctor



} // namespace dyablo

#endif // CONVERT_TO_PRIMITIVES_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
