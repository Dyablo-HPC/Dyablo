/**
 * \file ComputeDtHydroHashFunctor.h
 * \author Pierre Kestener
 * \date May 23rd 2020
 */
#ifndef COMPUTE_DT_HYDRO_MUSCL_BLOCK_HASH_FUNCTOR_H_
#define COMPUTE_DT_HYDRO_MUSCL_BLOCK_HASH_FUNCTOR_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/initRiemannConfig2d.h"

#include "shared/AMRMetaData.h"
#include "shared/amr/AMRmesh.h"

// hydro utils
#include "shared/utils_hydro.h"
#include "shared/morton_utils.h"

#include "utils_block.h"

namespace dyablo
{
namespace muscl_block
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Simplest CFL computational functor.
 * All cell, whatever level, contribute equally to the CFL condition.
 *
 */
template<int dim>
class ComputeDtHydroHashFunctor
{

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  // AMR related type alias
  using key_t = typename AMRMetaData<dim>::key_t;
  using value_t = typename AMRMetaData<dim>::value_t;

  // kokkos execution policy type alias
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  /**
   * \param[in] mes is an AMRMetaData object for connectivity
   */
  ComputeDtHydroHashFunctor(AMRMetaData<dim> mesh,
                            HydroParams    params,
                            id2index_t     fm,
                            blockSize_t    blockSizes,
                            DataArrayBlock Udata) :
    mesh(mesh), params(params),
    fm(fm), blockSizes(blockSizes),
    Udata(Udata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(AMRMetaData<dim> mesh,
		    ConfigMap        configMap,
                    HydroParams      params,
		    id2index_t       fm,
                    blockSize_t      blockSizes,
                    DataArrayBlock   Udata,
		    double          &invDt)
  {
    
    ComputeDtHydroHashFunctor functor(mesh, params, fm, blockSizes, Udata);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("amr","nbTeams",16);
    functor.setNbTeams ( nbTeams_ );

    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_reduce("dyablo::muscl_block::ComputeDtHydroHashFunctor",
                            policy, functor, invDt);
  } // apply

  // ====================================================================
  // ====================================================================
  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (real_t& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
  } // init

  // ====================================================================
  // ================================================
  // ================================
  // 2D version.
  // ================================
  // ================================================
  // ====================================================================
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==2, thread_t>::type& member,
                  real_t &invDt) const
  {
    uint32_t iOct = member.league_rank();
    //uint32_t iCell = member.team_rank();

    // get a const ref on the array of levels
    const auto levels = mesh.levels();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    // number of cells per octant
    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    // initialize reduction variable
    real_t invDt_local = invDt;

    // 2D version
    HydroState2d uLoc; // conservative variables in current cell
    HydroState2d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    real_t vx, vy;

    while (iOct < nbOcts) {

      // get octant level
      uint8_t level = levels(iOct);
      
      // retrieve cell size from mesh
      real_t dx = levelToSize(level) / blockSizes[IX];
      real_t dy = levelToSize(level) / blockSizes[IY];

      // initialialize cell id
      uint32_t iCell = member.team_rank();
      while (iCell < nbCells) {
        
        // get local conservative variable
        uLoc[ID] = Udata(iCell,fm[ID],iOct);
        uLoc[IP] = Udata(iCell,fm[IP],iOct);
        uLoc[IU] = Udata(iCell,fm[IU],iOct);
        uLoc[IV] = Udata(iCell,fm[IV],iOct);
        
        // get primitive variables in current cell
        computePrimitives(uLoc, &c, qLoc, params);

        if (params.rsst_enabled and params.rsst_cfl_enabled) {
          vx = c/params.rsst_ksi + FABS(qLoc[IU]);
          vy = c/params.rsst_ksi + FABS(qLoc[IV]);
        } else {
          vx = c + FABS(qLoc[IU]);
          vy = c + FABS(qLoc[IV]);
        }

        invDt_local = FMAX(invDt_local, vx / dx + vy / dy);

        iCell += member.team_size();
      
      } // end while iCell

      iOct += nbTeams;

    } // end while iOct

    // update global reduced value
    if (invDt < invDt_local)
      invDt = invDt_local;

  } // operator_2d

  // ====================================================================
  // ================================================
  // ================================
  // 2D version.
  // ================================
  // ================================================
  // ====================================================================
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==3, thread_t>::type& member,
                  real_t &invDt) const
  {
    uint32_t iOct = member.league_rank();
    uint32_t iCell = member.team_rank();

    // get a const ref on the array of levels
    const auto levels = mesh.levels();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    real_t invDt_local = invDt;
    
    // 3D version
    HydroState3d uLoc; // conservative variables in current cell
    HydroState3d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    real_t vx, vy, vz;

    while (iOct < nbOcts) {

      // get octant level
      uint8_t level = levels(iOct);
      
      // retrieve cell size from mesh
      real_t dx = levelToSize(level) / blockSizes[IX];
      real_t dy = levelToSize(level) / blockSizes[IY];
      real_t dz = levelToSize(level) / blockSizes[IZ];

      while (iCell < nbCells) {
    
        // get local conservative variable
        uLoc[ID] = Udata(iCell,fm[ID],iOct);
        uLoc[IP] = Udata(iCell,fm[IP],iOct);
        uLoc[IU] = Udata(iCell,fm[IU],iOct);
        uLoc[IV] = Udata(iCell,fm[IV],iOct);
        uLoc[IW] = Udata(iCell,fm[IW],iOct);
        
        // get primitive variables in current cell
        computePrimitives(uLoc, &c, qLoc, params);

        if (params.rsst_enabled and params.rsst_cfl_enabled) {
          vx = c/params.rsst_ksi + FABS(qLoc[IU]);
          vy = c/params.rsst_ksi + FABS(qLoc[IV]);
          vz = c/params.rsst_ksi + FABS(qLoc[IW]);
        } else {
          vx = c + FABS(qLoc[IU]);
          vy = c + FABS(qLoc[IV]);
          vz = c + FABS(qLoc[IW]);
        }

        invDt_local = FMAX(invDt_local, vx / dx + vy / dy + vz / dz);

        iCell += member.team_size();
      
      } // end while iCell

      iOct += nbTeams;

    } // end while iOct

    // update global reduced value
    if (invDt < invDt_local)
      invDt = invDt_local;

  } // operator_3d
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile real_t& dst,
	     const volatile real_t& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  //! AMR mesh
  AMRMetaData<dim> mesh;

  //! number of regular octants
  uint32_t         nbOcts;
  
  //! general parameters
  HydroParams      params;

  //! field manager
  id2index_t       fm;

  //! block sizes
  blockSize_t      blockSizes;

  //! heavy data - conservative variables
  DataArrayBlock   Udata;
  
}; // ComputeDtHydroHashFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // COMPUTE_DT_HYDRO_MUSCL_BLOCK_HASH_FUNCTOR_H_
