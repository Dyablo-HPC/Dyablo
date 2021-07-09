/**
 * @author Maxime Delorme
 **/

#pragma once

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"

// utils block
#include "muscl_block/utils_block.h"

namespace dyablo {
namespace muscl_block {

/**
 * Class filling up the gravitational data in case of a static gravitation
 **/
class GravityFillDataFunctor {
private:
  uint32_t nbTeams;

public:
  using index_t       = int32_t;
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<index_t>>;
  using thread_t      = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 


  /**
   * Constructor
   * 
   * \param[in] params parameters of the run
   * \param[in] fm field manager
   * \param[in] blockSizes size of the blocks 
   * \param[in] nbOcts total number of octants in the domain
   * \param[in] data_in global data to fill in
   **/
  GravityFillDataFunctor(LightOctree    lmesh,
                         ConfigMap      configMap,
                         HydroParams    params,
                         id2index_t     fm,
                         blockSize_t    blockSizes,
                         uint32_t       nbOcts,
                         DataArrayBlock data_in) :
    lmesh(lmesh),
    params(params),
    fm(fm),
    blockSizes(blockSizes),
    nbOcts(nbOcts),
    data_in(data_in) {
    nDim = (params.dimType == THREE_D ? 3 : 2);
  }

  /**
   * Builds and applies the functor to the current group
   * 
   * \param[in] lmesh the LightOctree structure holding the mesh
   * \param[in] configMap parameter reader
   * \param[in] params parameters of the run
   * \param[in] fm field manager
   * \param[in] blockSizes size of the blocks 
   * \param[in] nbOcts total number of octants in the domain
   * \param[in] data_in global data to fill in
   **/
  static void apply(LightOctree    lmesh,
                    ConfigMap      configMap,
                    HydroParams    params,
                    id2index_t     fm,
                    blockSize_t    blockSizes,
                    uint32_t       nbOcts,
                    DataArrayBlock data_in) 
  {
    // Building the functor
    GravityFillDataFunctor functor(lmesh,
                                   configMap,
                                   params,
                                   fm,
                                   blockSizes,
                                   nbOcts,
                                   data_in);

    // Setting up execution policy
    uint32_t nbTeams_ = configMap.getInteger("amr", "nbTeams", 16);
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_,
                         Kokkos::AUTO());

    Kokkos::parallel_for("dyablo::muscl_block::GravityFillDataFunctor",
                         policy, functor);
  }

  template<int ndims>
  KOKKOS_INLINE_FUNCTION
  void fill_gravity(thread_t member) const {
    uint32_t iOct = member.league_rank();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];
    uint32_t nbCells = (ndims == 2 ? bx*by : bx*by*bz);

    while (iOct < nbOcts) {
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        [&](const index_t index) {
          coord_t cell_coord = index_to_coord<ndims>(index, blockSizes);

          data_in(index, fm[IGX], iOct) = params.gx;
          data_in(index, fm[IGY], iOct) = params.gy;
          if (ndims == 3)
            data_in(index, fm[IGZ], iOct) = params.gz;
        });

      iOct   += nbTeams;
    }
  }

  /**
   * Functor operator
   * @param member the current thread team
   **/
  KOKKOS_INLINE_FUNCTION
  void operator()(thread_t member) const {
    if (nDim == 2)
      fill_gravity<2>(member);
    else
      fill_gravity<3>(member);
  }

  // Attributes
  LightOctree    lmesh;      //!< Mesh data structure
  HydroParams    params;     //!< Run parameters
  id2index_t     fm;         //!< Field manager
  blockSize_t    blockSizes; //!< Size of the blocks
  uint32_t       nbOcts;     //!< Number of octants
  DataArrayBlock data_in;    //!< Data array to fill
  int            nDim;       //!< Number of dimensions
}; // GravityFillDataFunctor
}
} 
