/**
 * \file CopyCornerBlockCellData.h
 * \author Maxime Delorme
 */
#ifndef COPY_CORNER_BLOCK_CELL_DATA_FUNCTOR_H_
#define COPY_CORNER_BLOCK_CELL_DATA_FUNCTOR_H_

#include "shared/FieldManager.h"
#include "shared/kokkos_shared.h"
#include "shared/utils_hydro.h"
#include "muscl_blolck/utils_block.h"

namespace dyablo {
namespace muscl_block {

  /***************************************/
  /**
   * TODO : Documentation
   **/

class CopyCornerBlockCellDataFunctor {
private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using index_t = int32_t;
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<index_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  CopyCornerBlockCellDataFunctor(std::shared_ptr<AMRmesh> pmesh,
				 HydroParams parms,
				 id2index_t fm,
				 blockSize_t blockSizes,
				 uint32_t ghostWidth,
				 uint32_t nbOctsPerGroup,
				 DataArrayBlock U,
				 DataArrayBlock Ugroup,
				 uint32_t iGroup) :
    pmesh(pmesh),
    params(params),
    fm(fm)
    blockSizes(blockSizes),
    ghosstWidth(ghosstWidth),
    nbOctsPerGroup(nbOctsPerGroup),
    U(U),
    U_ghost(U_ghost),
    Ugroup(Ugroup),
    iGroup(iGroup)
  {
    
  }

  // static method that does everything: creates and executes the functor
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
                    uint32_t iGroup,
                    FlagArrayBlock Interface_flags)
  {

    CopyCornerBlockCellDataFunctor functor(pmesh, params, fm, 
					   blockSizes, ghostWidth,
					   nbOctsPerGroup, 
					   U, U_ghost, 
					   Ugroup, iGroup,
					   Interface_flags);

    /*
     * using kokkos team execution policy
     */
    uint32_t nbTeams_ = configMap.getInteger("amr", "nbTeams", 16);
    functor.setNbTeams(nbTeams_);

    // create execution policy
    team_policy_t policy(nbTeams_,
                         Kokkos::AUTO() /* team size chosen by kokkos */);

    // launch computation (parallel kernel)
    Kokkos::parallel_for("dyablo::muscl_block::CopyCornerBlockCellDataFunctor",
                         policy, functor);
  }

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

}; // CopyCornerBlockCellDataFunctor

} // namespace muscl_block
} // namespace dyablo

#endif
