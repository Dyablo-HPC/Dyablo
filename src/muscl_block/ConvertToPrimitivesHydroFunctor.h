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
#include "shared/bitpit_common.h"

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
  ConvertToPrimitivesHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
				  HydroParams    params,
				  id2index_t     fm,
                                  blockSize_t    blockSizes,
				  DataArrayBlock Udata,
				  DataArrayBlock Qdata) :
    pmesh(pmesh), params(params),
    fm(fm), blockSizes(blockSizes), 
    Udata(Udata), Qdata(Qdata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap      configMap,
		    HydroParams    params,
		    id2index_t     fm,
                    blockSize_t    blockSizes,
		    DataArrayBlock Udata,
                    DataArrayBlock Qdata)
  {

    ConvertToPrimitivesHydroFunctor functor(pmesh, params, fm, 
                                            blockSizes,
                                            Udata, Qdata);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("amr","nbTeams",16);
    functor.setNbTeams ( nbTeams_ );
    
    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);
    
    
    Kokkos::parallel_for("dyablo::muscl_block::ConvertToPrimitivesHydroFunctor",
                         policy, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator_2d(team_policy_t::member_type member) const
  {
    
    uint32_t iOct = member.league_rank();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    while (iOct <  pmesh->getNumOctants() )
    {

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        KOKKOS_LAMBDA(const int32_t index) {

          HydroState2d uLoc; // conservative variables in current cell
          HydroState2d qLoc; // primitive    variables in current cell
          real_t c = 0.0;

          // get local conservative variable
          uLoc[ID] = Udata(index,fm[ID],iOct);
          uLoc[IP] = Udata(index,fm[IP],iOct);
          uLoc[IU] = Udata(index,fm[IU],iOct);
          uLoc[IV] = Udata(index,fm[IV],iOct);
    
          // get primitive variables in current cell
          computePrimitives(uLoc, &c, qLoc, params);
          
          // copy q state in q global
          Qdata(index,fm[ID],iOct) = qLoc[ID];
          Qdata(index,fm[IP],iOct) = qLoc[IP];
          Qdata(index,fm[IU],iOct) = qLoc[IU];
          Qdata(index,fm[IV],iOct) = qLoc[IV];

        }); // end TeamVectorRange

      iOct += nbTeams;
      
    } // end while iOct < nbOct

  } // operator_2d

  KOKKOS_INLINE_FUNCTION
  void operator_3d(team_policy_t::member_type member) const
  {
    
    uint32_t iOct = member.league_rank();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    while (iOct <  pmesh->getNumOctants() )
    {

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        KOKKOS_LAMBDA(const int32_t index) {
    
          HydroState3d uLoc; // conservative variables in current cell
          HydroState3d qLoc; // primitive    variables in current cell
          real_t c = 0.0;
    
          // get local conservative variable
          uLoc[ID] = Udata(index,fm[ID],iOct);
          uLoc[IP] = Udata(index,fm[IP],iOct);
          uLoc[IU] = Udata(index,fm[IU],iOct);
          uLoc[IV] = Udata(index,fm[IV],iOct);
          uLoc[IW] = Udata(index,fm[IW],iOct);
          
          // get primitive variables in current cell
          computePrimitives(uLoc, &c, qLoc, params);
          
          // copy q state in q global
          Qdata(index,fm[ID],iOct) = qLoc[ID];
          Qdata(index,fm[IP],iOct) = qLoc[IP];
          Qdata(index,fm[IU],iOct) = qLoc[IU];
          Qdata(index,fm[IV],iOct) = qLoc[IV];
          Qdata(index,fm[IW],iOct) = qLoc[IW];
        
        }); // end TeamVectorRange

      iOct += nbTeams;

    } // end while iOct < nbOct

  } // operator_3d

  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const
  {
    
    if (params.dimType == TWO_D)
      operator_2d(member);
    
    if (params.dimType == THREE_D)
      operator_3d(member);
    
  } // operator ()
  
  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;
    
  //! general parameters
  HydroParams  params;
  
  //! field manager
  id2index_t   fm;

  //! block sizes
  blockSize_t blockSizes;

  //! heavy data - conservative variables
  DataArrayBlock    Udata;

  //! heavy data - primitive variables
  DataArrayBlock    Qdata;

}; // ConvertToPrimitivesHydroFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // CONVERT_TO_PRIMITIVES_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
