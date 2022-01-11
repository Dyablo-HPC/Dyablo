/**
 * \file ConvertToPrimitivesHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef CONVERT_TO_PRIMITIVES_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
#define CONVERT_TO_PRIMITIVES_HYDRO_MUSCL_BLOCK_FUNCTOR_H_

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/amr/AMRmesh.h"

// utils hydro
#include "shared/utils_hydro.h"

// utils block
#include "muscl_block/utils_block.h"

namespace dyablo { namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Convert conservative variables to primitive variables functor.
 *
 */
class ConvertToPrimitivesHydroFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  /**
   * Convert conservative variables to primitive ones using equation of state.
   *
   * \param[in] params
   * \param[in] Udata conservative variables
   * \param[out] Qdata primitive variables
   */
  ConvertToPrimitivesHydroFunctor(HydroParams    params,
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
    
    nbCellsPerBlock = params.dimType == TWO_D ? 
      bx_g * by_g :
      bx_g * by_g * bz_g;

  };
  
  // static method which does it all: create and execute functor
  static void apply(ConfigMap      configMap,
                    HydroParams    params,
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

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getValue<uint32_t>("amr","nbTeams",16);
    functor.setNbTeams ( nbTeams_ );
    
    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);
    
    
    Kokkos::parallel_for("dyablo::muscl_block::ConvertToPrimitivesHydroFunctor",
                         policy, functor);
  }

  // ========================================================================
  // ========================================================================
  KOKKOS_INLINE_FUNCTION
  void cons2prim_2d(const int32_t index,
                    const uint32_t iOct_local) const
  {
  
    HydroState2d uLoc; // conservative variables in current cell
    HydroState2d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    
    // get local conservative variable
    uLoc[ID] = Ugroup(index,fm[ID],iOct_local);
    uLoc[IP] = Ugroup(index,fm[IP],iOct_local);
    uLoc[IU] = Ugroup(index,fm[IU],iOct_local);
    uLoc[IV] = Ugroup(index,fm[IV],iOct_local);
    
    // get primitive variables in current cell
    computePrimitives(uLoc, &c, qLoc, params);
    
    // copy q state in q global
    Qgroup(index,fm[ID],iOct_local) = qLoc[ID];
    Qgroup(index,fm[IP],iOct_local) = qLoc[IP];
    Qgroup(index,fm[IU],iOct_local) = qLoc[IU];
    Qgroup(index,fm[IV],iOct_local) = qLoc[IV];
    
  } // cons2prim_2d

  // ========================================================================
  // ========================================================================
  KOKKOS_INLINE_FUNCTION
  void cons2prim_3d(const int32_t index,
                    const uint32_t iOct_local) const
  {
  
    HydroState3d uLoc; // conservative variables in current cell
    HydroState3d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    
    // get local conservative variable
    uLoc[ID] = Ugroup(index,fm[ID],iOct_local);
    uLoc[IP] = Ugroup(index,fm[IP],iOct_local);
    uLoc[IU] = Ugroup(index,fm[IU],iOct_local);
    uLoc[IV] = Ugroup(index,fm[IV],iOct_local);
    uLoc[IW] = Ugroup(index,fm[IW],iOct_local);
    
    // get primitive variables in current cell
    computePrimitives(uLoc, &c, qLoc, params);
    
    // copy q state in q global
    Qgroup(index,fm[ID],iOct_local) = qLoc[ID];
    Qgroup(index,fm[IP],iOct_local) = qLoc[IP];
    Qgroup(index,fm[IU],iOct_local) = qLoc[IU];
    Qgroup(index,fm[IV],iOct_local) = qLoc[IV];
    Qgroup(index,fm[IW],iOct_local) = qLoc[IW];
    
  } // cons2prim_3d

  // ========================================================================
  // ========================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const
  {

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    while (iOct < iOctNextGroup and iOct < nbOcts)
    {

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock),
        [&](const int32_t index) {

          if (params.dimType == TWO_D)
            cons2prim_2d(index, iOct_local);

          else if (params.dimType == THREE_D)
            cons2prim_3d(index, iOct_local);
  
        }); // end TeamVectorRange

      iOct       += nbTeams;
      iOct_local += nbTeams;

    } // end while iOct < nbOct

  } // operator()

  //! general parameters
  HydroParams  params;
  
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

} // namespace muscl_block

} // namespace dyablo

#endif // CONVERT_TO_PRIMITIVES_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
