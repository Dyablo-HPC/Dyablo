/**
 * \file InitBlast.h
 * \author Pierre Kestener
 */
#ifndef MUSCL_BLOCK_HYDRO_INIT_BLAST_H_
#define MUSCL_BLOCK_HYDRO_INIT_BLAST_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/BlastParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace dyablo { namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve blast problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 * 
 *
 * Initial conditions is refined near strong density gradients.
 *
 * Using Kokkos Team policy, one thread team per block (one block per octree leaf).
 *
 */
class InitBlastDataFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  InitBlastDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                       HydroParams    params,
                       BlastParams    bParams,
                       id2index_t     fm,
                       Kokkos::Array<int,3> blockSizes,
                       DataArrayBlock Udata) :
    pmesh(pmesh), params(params), bParams(bParams),
    fm(fm), blockSizes(blockSizes), Udata(Udata)
  {
  };
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams    params,
                    ConfigMap      configMap,
		    id2index_t     fm,
                    Kokkos::Array<int32_t,3> blockSizes,
                    DataArrayBlock Udata)
  {
    BlastParams blastParams = BlastParams(configMap);
    
    // data init functor
    InitBlastDataFunctor functor(pmesh, params, blastParams, fm, blockSizes, Udata);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("init","nbTeams",16);
    functor.setNbTeams ( nbTeams_  );

    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("dyablo::muscl_block::InitBlastDataFunctor",
                         policy, functor);
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const
  {

    // blast problem parameters
    const real_t blast_radius      = bParams.blast_radius;
    const real_t radius2           = blast_radius*blast_radius;
    const real_t blast_center_x    = bParams.blast_center_x;
    const real_t blast_center_y    = bParams.blast_center_y;
    const real_t blast_center_z    = bParams.blast_center_z;
    const real_t blast_density_in  = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out= bParams.blast_pressure_out;
    
    const real_t gamma0            = params.settings.gamma0;

    uint32_t iOct = member.league_rank();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    while (iOct <  pmesh->getNumOctants() )
    {
      
      // get octant size
      const real_t octSize = pmesh->getSize(iOct);

      // the following assumes bx=by=bz
      // TODO : allow non-cubic blocks
      const real_t cellSize = octSize/bx;

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      // std::array<double,3> center = pmesh->getCenter(iOct);
      
      // const real_t xc = center[0];
      // const real_t yc = center[1];
      // const real_t zc = center[2];

      // coordinates of the lower left corner
      const real_t x0 = pmesh->getNode(iOct, 0)[IX];
      const real_t y0 = pmesh->getNode(iOct, 0)[IY];
      const real_t z0 = pmesh->getNode(iOct, 0)[IZ];

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        KOKKOS_LAMBDA(const int32_t index) {

          // convert index into ix,iy,iz of local cell inside
          // block
          Kokkos::Array<int32_t,3> iCoord;
          int32_t& ix = iCoord[IX];
          int32_t& iy = iCoord[IY];
          int32_t& iz = iCoord[IZ];                    

          if (params.dimType == TWO_D) {
            iCoord = compute_cell_coord(index,bx,by,1);
          } else {
            iCoord = compute_cell_coord(index,bx,by,bz);
          }

          // compute x,y,z for that cell (cell center)
          real_t x = x0 + ix*cellSize + cellSize/2;
          real_t y = y0 + iy*cellSize + cellSize/2;
          real_t z = z0 + iz*cellSize + cellSize/2;

          // initialize
          real_t d2 = 
            (x-blast_center_x)*(x-blast_center_x)+
            (y-blast_center_y)*(y-blast_center_y);  
          
          if (params.dimType == THREE_D)
            d2 += (z-blast_center_z)*(z-blast_center_z);
          
          if (d2 < radius2) {
            Udata(index, fm[ID], iOct) = blast_density_in;
            Udata(index, fm[IP], iOct) = blast_pressure_in/(gamma0-1.0);
            Udata(index, fm[IU], iOct) = 0.0;
            Udata(index, fm[IV], iOct) = 0.0;
          } else {
            Udata(index, fm[ID], iOct) = blast_density_out;
            Udata(index, fm[IP], iOct) = blast_pressure_out/(gamma0-1.0);
            Udata(index, fm[IU], iOct) = 0.0;
            Udata(index, fm[IV], iOct) = 0.0;
          }

          if (params.dimType == THREE_D)
            Udata(index, fm[IW], iOct) = 0.0;

        }); // end TeamVectorRange

      iOct += nbTeams;

    } // end while

  } // end operator ()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams    params;

  //! Blast problem specific parameters
  BlastParams    bParams;

  // field manager
  id2index_t     fm;

  //! block sizes
  Kokkos::Array<int32_t, 3> blockSizes;

  //! heavy data
  DataArrayBlock Udata;

}; // InitBlastDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve blast problem.
 *
 * This functor only performs mesh refinement, no user data init.
 * 
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitBlastDataFunctor
 *
 */
class InitBlastRefineFunctor {
  
public:
  InitBlastRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                         HydroParams  params,
                         BlastParams bParams,
                         int         level_refine) :
    pmesh(pmesh), params(params), bParams(bParams),
    level_refine(level_refine)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap     configMap,
		    HydroParams   params,
		    int           level_refine)
  {
    BlastParams blastParams = BlastParams(configMap);

    // iterate functor for refinement
    InitBlastRefineFunctor functor(pmesh, params, blastParams, 
                                   level_refine);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
    
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {

    // blast problem parameters
    const real_t radius            = bParams.blast_radius;
    //const real_t radius2           = radius*radius;
    const real_t blast_center_x    = bParams.blast_center_x;
    const real_t blast_center_y    = bParams.blast_center_y;
    const real_t blast_center_z    = bParams.blast_center_z;

    //constexpr double eps = 0.005;
    
    // get cell level
    uint8_t level = pmesh->getLevel(i);
    
    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double,3> center = pmesh->getCenter(i);
      
      const real_t x = center[0];
      const real_t y = center[1];
      const real_t z = center[2];

      double cellSize2 = pmesh->getSize(i)*0.75;
      
      bool should_refine = false;

      real_t d2 = 
        (x-blast_center_x)*(x-blast_center_x)+
        (y-blast_center_y)*(y-blast_center_y);  

      if (params.dimType == THREE_D)
        d2 += (z-blast_center_z)*(z-blast_center_z);

      if ( fabs(sqrt(d2) - radius) < cellSize2 )
	should_refine = true;
      
      if (should_refine)
	pmesh->setMarker(i, 1);

    } // end if level == level_refine
    
  } // end operator ()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams    params;

  //! Blast problem specific parameters
  BlastParams    bParams;

  //! block sizes
  //Kokkos::Array<int, 3>  blockSizes;

  //! which level should we look at
  int            level_refine;
  
}; // InitBlastRefineFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_HYDRO_INIT_BLAST_H_
