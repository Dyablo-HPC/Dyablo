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
#include "shared/amr/AMRmesh.h"

namespace dyablo { namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve blast problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data on host. Copying data from host to device, should be 
 * done outside.
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
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  InitBlastDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                       HydroParams    params,
                       BlastParams    bParams,
                       id2index_t     fm,
                       blockSize_t    blockSizes,
                       DataArrayBlockHost Udata_h) :
    pmesh(pmesh), params(params), bParams(bParams),
    fm(fm), blockSizes(blockSizes), Udata_h(Udata_h)
  {
  };
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		                HydroParams    params,
                    ConfigMap      configMap,
		                id2index_t     fm,
                    blockSize_t    blockSizes,
                    DataArrayBlockHost Udata_h)
  {
    BlastParams blastParams = BlastParams(configMap);
    
    // data init functor
    InitBlastDataFunctor functor(pmesh, params, blastParams, fm, blockSizes, Udata_h);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getValue<uint32_t>("init","nbTeams",16);
    functor.setNbTeams ( nbTeams_  );

    // perform initialization on host
    team_policy_t policy (Kokkos::OpenMP(),
                          nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("dyablo::muscl_block::InitBlastDataFunctor",
                         policy, functor);
  }
  
  //KOKKOS_INLINE_FUNCTION
  inline
  void operator()(thread_t member) const
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
      const real_t x0 = pmesh->getCoordinates(iOct)[IX];
      const real_t y0 = pmesh->getCoordinates(iOct)[IY];
      const real_t z0 = pmesh->getCoordinates(iOct)[IZ];

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        [=](const int32_t index) {

          // convert index into ix,iy,iz of local cell inside
          // block
          coord_t iCoord;
          uint32_t& ix = iCoord[IX];
          uint32_t& iy = iCoord[IY];
          uint32_t& iz = iCoord[IZ];                    

          if (params.dimType == TWO_D) {
            iCoord = index_to_coord<2>(index,blockSizes);
          } else {
            iCoord = index_to_coord<3>(index,blockSizes);
          }

          // compute x,y,z for that cell (cell center)
          real_t x = x0 + ix*cellSize + cellSize/2;
          real_t y = y0 + iy*cellSize + cellSize/2;
          real_t z = z0 + iz*cellSize + cellSize/2;

          // Quadrant size
          const real_t qx = 1.0 / bParams.blast_nx;
          const real_t qy = 1.0 / bParams.blast_ny;
          const real_t qz = 1.0 / bParams.blast_nz;
          const real_t q = std::min({qx, qy, qz});
          
          const int qix = (int)(x / qx);
          const int qiy = (int)(y / qy);
          const int qiz = (int)(z / qz);

          // Rescaling position wrt the current blast quadrant
          x = (x - qix * qx) / q - (qx/q - 1) * 0.5;
          y = (y - qiy * qy) / q - (qy/q - 1) * 0.5;
          z = (z - qiz * qz) / q - (qz/q - 1) * 0.5;

          // initialize
          real_t d2 = 
            (x-blast_center_x)*(x-blast_center_x)+
            (y-blast_center_y)*(y-blast_center_y);  
          
          if (params.dimType == THREE_D)
            d2 += (z-blast_center_z)*(z-blast_center_z);
          
          if (d2 < radius2) {
            Udata_h(index, fm[ID], iOct) = blast_density_in;
            Udata_h(index, fm[IP], iOct) = blast_pressure_in/(gamma0-1.0);
            Udata_h(index, fm[IU], iOct) = 0.0;
            Udata_h(index, fm[IV], iOct) = 0.0;
          } else {
            Udata_h(index, fm[ID], iOct) = blast_density_out;
            Udata_h(index, fm[IP], iOct) = blast_pressure_out/(gamma0-1.0);
            Udata_h(index, fm[IU], iOct) = 0.0;
            Udata_h(index, fm[IV], iOct) = 0.0;
          }

          if (params.dimType == THREE_D)
            Udata_h(index, fm[IW], iOct) = 0.0;

        }); // end TeamVectorRange

      iOct += nbTeams;

    } // end while

  } // end operator ()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams        params;

  //! Blast problem specific parameters
  BlastParams        bParams;

  //! field manager
  id2index_t         fm;

  //! block sizes
  blockSize_t        blockSizes;

  //! heavy data on host
  DataArrayBlockHost Udata_h;

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
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;

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
    
    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());
    Kokkos::parallel_for("dyablo::muscl_block::InitBlastRefineFunctor",
                         policy, functor);
    
  } // apply
  
  //KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& iOct) const
  {

    // blast problem parameters
    const real_t radius            = bParams.blast_radius;
    //const real_t radius2           = radius*radius;
    const real_t blast_center_x    = bParams.blast_center_x;
    const real_t blast_center_y    = bParams.blast_center_y;
    const real_t blast_center_z    = bParams.blast_center_z;

    //constexpr double eps = 0.005;
    
    // get cell level
    uint8_t level = pmesh->getLevel(iOct);
    
    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double,3> center = pmesh->getCenter(iOct);
      
      real_t x = center[0];
      real_t y = center[1];
      real_t z = center[2];

      // Quadrant size
      const real_t qx = 1.0 / bParams.blast_nx;
      const real_t qy = 1.0 / bParams.blast_ny;
      const real_t qz = (params.dimType == THREE_D ? 1.0 / bParams.blast_nz : 1.0);
      const real_t q = std::min({qx, qy, qz});

      const int qix = (int)(x / qx);
      const int qiy = (int)(y / qy);
      const int qiz = (int)(z / qz);

      // Rescaling position wrt the current blast quadrant
      x = (x - qix * qx) / q - (qx/q - 1) * 0.5;
      y = (y - qiy * qy) / q - (qy/q - 1) * 0.5;
      z = (z - qiz * qz) / q - (qz/q - 1) * 0.5;

      // Two refinement criteria are used : 
      //  1- If the cell size is larger than a quadrant we refine
      //  2- If the distance to the blast is smaller than the size of
      //     half a diagonal we refine

      real_t cellSize = pmesh->getSize(iOct);
      
      bool should_refine = (cellSize > std::min({qx, qy, qz}));

      real_t d2 = std::pow(x - blast_center_x, 2) +
                  std::pow(y - blast_center_y, 2);
      if (params.dimType == THREE_D)
        d2 += std::pow(z - blast_center_z, 2);

      // Cell diag is calculated to be in the units of a quadrant
      const real_t cx = cellSize / qx;
      const real_t cy = cellSize / qy;
      const real_t cz = cellSize / qz;

      real_t cellDiag = (params.dimType == THREE_D 
                          ? sqrt(cx*cx+cy*cy+cz*cz) * 0.5
                          : sqrt(cx*cx+cy*cy) * 0.5);

      if (fabs(sqrt(d2) - radius) < cellDiag)
        should_refine = true; 
      
      if (should_refine)
	      pmesh->setMarker(iOct, 1);

    } // end if level == level_refine
    
  } // end operator ()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams    params;

  //! Blast problem specific parameters
  BlastParams    bParams;

  //! block sizes
  //blockSize_t  blockSizes;

  //! which level should we look at
  int            level_refine;
  
}; // InitBlastRefineFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_HYDRO_INIT_BLAST_H_
