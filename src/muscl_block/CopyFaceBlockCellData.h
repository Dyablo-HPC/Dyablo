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
 * The main difficulty here is to deploy the entire combinatorics of
 * geometrical possibilities in terms of 
 * - size of neighbor octant, i.e.
 *   is neighbor octant small, same size or larger thant current octant,
 * - direction : face along X, Y or Z behave slightly differently, need to
 *   efficiently take symetries into account
 * - 2d/3d
 *
 * So we need to be careful, have good testing code.
 * See file test_CopyGhostBlockCellData.cpp
 *
 * \note As always use the nested parallelism strategy:
 * - loop over octants              parallelized with Team policy,
 * - loop over cells inside a block paralellized with ThreadVectorRange policy.
 *
 *
 * In reality, to simplify things, we assume block sizes are even integers (TBC, maybe no needed).
 *
 * \sa functor CopyInnerBlockCellDataFunctor
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

  /**
   * enum used to identify the relative position of an octant
   * versus a face neighbor octant of larger or smaller size.
   *
   * In 2d, only 2 possibilities
   * In 3d, there are 4 possibilities
   */
  enum NEIGH_LOC : uint8_t {
    NEIGH_POS_0 = 0,
    NEIGH_POS_1 = 1,
    NEIGH_POS_2 = 2,
    NEIGH_POS_3 = 3
  };

  enum NEIGH_SIZE : uint8_t {
    NEIGH_IS_SMALLER   = 0,
    NEIGH_IS_LARGER    = 1,
    NEIGH_IS_SAME_SIZE = 2
  };

  /**
   *
   * \param[in] pmesh the main PABLO data structure with AMR connectivity information
   * \param[in] params hydrodynamics parameters (geometry, ...)
   * \param[in] fm field manager to access applicative/hydrodynamics variables
   * \param[in] blockSizes x,y,z sizes of block of data per octant (no ghost)
   * \param[in] ghostWidth number (width) of ghost cell arround block
   * \param[in] number of octants per group
   * \param[in] U conservative variables - global block array data (no ghost)
   * \param[in] Ughost same as U but for MPI ghost octants
   * \param[out] Ugroup conservative var of a group of octants (block data with
   *             ghosts) to be used later in application / numerical scheme
   * \param[in] iGroup identify the group of octant we want to copy
   *
   * We probably could avoid passing a HydroParams object. // Refactor me ?
   *
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
                               DataArrayBlock Gravity,
                               DataArrayBlock Gravity_ghost,
                               DataArrayBlock Ggroup,
                               uint32_t iGroup,
                               FlagArrayBlock Interface_flags) :
    pmesh(pmesh), params(params), 
    fm(fm), blockSizes(blockSizes), 
    ghostWidth(ghostWidth),
    nbOctsPerGroup(nbOctsPerGroup), 
    U(U), 
    U_ghost(U_ghost),
    Ugroup(Ugroup), 
    Gravity(Gravity),
    Gravity_ghost(Gravity_ghost),
    Ggroup(Ggroup),
    iGroup(iGroup),
    Interface_flags(Interface_flags)
  {

    // in 2d, bz and bz_g are not used

    bx_g = blockSizes[IX] + 2 * ghostWidth;
    by_g = blockSizes[IY] + 2 * ghostWidth;
    bz_g = blockSizes[IZ] + 2 * ghostWidth;

    // Gravity
    copy_gravity = (params.gravity_type == GRAVITY_CST_FIELD);
    ndim = (params.dimType == THREE_D ? 3 : 2);
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
                    DataArrayBlock Gravity,
                    DataArrayBlock Gravity_ghost,
                    DataArrayBlock Ggroup,
                    uint32_t iGroup,
                    FlagArrayBlock Interface_flags)
  {

    CopyFaceBlockCellDataFunctor functor(pmesh, params, fm, 
                                         blockSizes, ghostWidth,
                                         nbOctsPerGroup, 
                                         U, U_ghost, Ugroup,
                                         Gravity, Gravity_ghost, Ggroup,
                                         iGroup,
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
    Kokkos::parallel_for("dyablo::muscl_block::CopyFaceBlockCellDataFunctor",
                         policy, functor);
  }

  // ==============================================================
  // ==============================================================
  /**
   * Identifies situations like this (assuming neighbor octant is on the left,
   * and current octant on the right) :
   *
   * \return eitheir NEIGH_POS_0 or NEIGH_POS_1 (see below)
   *
   * Algorithm to evaluate realtive position (see drawings below)
   * just consists in comparing X or Y coordinate of the lower
   * left corner of current octant and neighbor octant.
   *
   * assuming neighbor octant is LARGER than current octant
   *
   * ============================================
   * E.g. when dir = DIR_X and face = FACE_LEFT :
   *
   * NEIGH_POS_0          NEIGH_POS_1
   *  ______               ______    __
   * |      |             |      |  |  |
   * |      |   __    or  |      |  |__|
   * |      |  |  |       |      |
   * |______|  |__|       |______|
   *
   *
   * ============================================
   * E.g. when dir = DIR_Y and face = FACE_LEFT (i.e. below):
   *
   * NEIGH_POS_0          NEIGH_POS_1
   *  ______               ______ 
   * |      |             |      |
   * |      |         or  |      |
   * |      |             |      |
   * |______|             |______|
   *
   *  __                       __
   * |  |                     |  |
   * |__|                     |__|
   *
   * 
   * assuming neighbor octant is SMALLER than current octant
   *
   * ============================================
   * E.g. when dir = DIR_X and face = FACE_LEFT :
   *
   * NEIGH_POS_0            NEIGH_POS_1
   *        ______           __      ______ 
   *       |      |         |  |    |      |
   *  __   |      |     or  |__|    |      |
   * |  |  |      |                 |      |
   * |__|  |______|                 |______|
   *
   *
   * \note Please note that the "if" branches are not divergent,
   * all threads in a given team always follow the same
   * branch.
   */
  KOKKOS_INLINE_FUNCTION
  NEIGH_LOC get_relative_position_2d(uint32_t iOct,
                                     uint32_t iOct_neigh,
                                     bool     is_ghost,
                                     DIR_ID   dir,
                                     FACE_ID  face,
                                     NEIGH_SIZE neigh_size) const
  {

    // default value
    NEIGH_LOC res = NEIGH_POS_0;

    // the following is a bit dirty, because current PABLO does not allow
    // the user to probe logical coordinates (integer), only physical
    // coordinates (double)

    /*
     * check if we are dealing with face along X, Y or Z direction
     */
    if (dir == DIR_X) {

      // get Y coordinates of the lower left corner of current octant
      real_t cur_loc = pmesh->getY(iOct);

      // get Y coordinates of the lower left corner of neighbor octant
      real_t neigh_loc = is_ghost ? 
        pmesh->getYghost(iOct_neigh) : 
        pmesh->getY     (iOct_neigh);

      if (neigh_size == NEIGH_IS_LARGER  and (neigh_loc < cur_loc) )
        res = NEIGH_POS_1;

      if (neigh_size == NEIGH_IS_SMALLER and (neigh_loc > cur_loc) )
        res = NEIGH_POS_1;

    }

    if (dir == DIR_Y) {

      // get X coordinates of the lower left corner of current octant
      real_t cur_loc = pmesh->getX(iOct);

      // get X coordinates of the lower left corner of neighbor octant
      real_t neigh_loc = is_ghost ? 
        pmesh->getXghost(iOct_neigh) : 
        pmesh->getX     (iOct_neigh);

      if ( neigh_size == NEIGH_IS_LARGER and (neigh_loc < cur_loc) )
        res = NEIGH_POS_1;

      if ( neigh_size == NEIGH_IS_SMALLER and (neigh_loc > cur_loc) )
        res = NEIGH_POS_1;

    }

    return res;

  } // get_relative_position_2d

  // ==============================================================
  // ==============================================================
  /**
   * Fill ghost cells for octant touching an external border.
   *
   * \note PERIODIC border condition if already taken into account
   * when dealing with neighbor of same size.
   *
   */
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_face_2d_external_border(uint32_t iOct,
                                          uint32_t iOct_local,
                                          index_t  index_in,
                                          DIR_ID   dir,
                                          FACE_ID  face) const
  {

    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    const index_t size_borderX = ghostWidth*by;
    const index_t size_borderY = bx*ghostWidth; 
    
    // make sure index is valid, i.e. inside the range of admissible values
    if ((index_in < size_borderX and dir == DIR_X) or
        (index_in < size_borderY and dir == DIR_Y)) {

      // in case of a X border
      // border sizes for input  cell are : ghostWidth,by
      // border sizes for output cell are : ghostWidth,by+2*ghostWidth

      // in case of a Y border
      // border sizes for input  cell are : bx             ,ghostWidth
      // border sizes for output cell are : bx+2*ghostWidth,ghostWidth

      // compute cell coordinates inside border
      coord_t coord_cur = dir == DIR_X ? 
        index_to_coord(index_in, ghostWidth, by) :
        index_to_coord(index_in, bx, ghostWidth) ;
      
      // shift coord to face center
      if (dir == DIR_X)
        coord_cur[IY] += ghostWidth;
      if (dir == DIR_Y)
        coord_cur[IX] += ghostWidth;

      if ( face == FACE_RIGHT ) {
        // if necessary shift coordinate to the right
        if (dir == DIR_X)
          coord_cur[IX] += (bx+ghostWidth);
        if (dir == DIR_Y)
          coord_cur[IY] += (by+ghostWidth);
      }
      
      // compute corresponding index in the ghosted block (current octant) 
      uint32_t index_cur = coord_cur[IX] + (bx+2*ghostWidth)*coord_cur[IY];

      // normal momentum sign
      real_t sign_u = 1.0;
      real_t sign_v = 1.0;
      //real_t sign_w = 1.0;

      coord_t coord_in = {coord_cur[IX],
                          coord_cur[IY],
                          0};

      // absorbing border : we just copy/mirror data 
      // from the last inner cells into ghost cells
      if ( (params.boundary_type_xmin == BC_ABSORBING and face == FACE_LEFT) or
           (params.boundary_type_xmax == BC_ABSORBING and face == FACE_RIGHT) or
           (params.boundary_type_ymin == BC_ABSORBING and face == FACE_LEFT) or
           (params.boundary_type_ymax == BC_ABSORBING and face == FACE_RIGHT) ) 
      {

        if (dir == DIR_X and face == FACE_LEFT)
          coord_in[IX] = ghostWidth;
        if (dir == DIR_X and face == FACE_RIGHT)
          coord_in[IX] = bx + ghostWidth - 1;
        if (dir == DIR_Y and face == FACE_LEFT)
          coord_in[IY] = ghostWidth;
        if (dir == DIR_Y and face == FACE_RIGHT)
          coord_in[IY] = by + ghostWidth - 1;
      } // end ABSORBING


      // reflecting border : we just copy/mirror data 
      // from inner cells into ghost cells, and invert
      // normal momentum
      if ( (params.boundary_type_xmin == BC_REFLECTING and face == FACE_LEFT) or
           (params.boundary_type_xmax == BC_REFLECTING and face == FACE_RIGHT) or
           (params.boundary_type_ymin == BC_REFLECTING and face == FACE_LEFT) or
           (params.boundary_type_ymax == BC_REFLECTING and face == FACE_RIGHT) ) 
      {

        if (dir == DIR_X and face == FACE_LEFT) {
          coord_in[IX] = 2 * ghostWidth - 1 - coord_cur[IX];
          sign_u = -1.0;
        }
        if (dir == DIR_X and face == FACE_RIGHT) {
          coord_in[IX] = 2 * bx + 2 * ghostWidth - 1 - coord_cur[IX];
          sign_u = -1.0;
        }
        if (dir == DIR_Y and face == FACE_LEFT) {
          coord_in[IY] = 2 * ghostWidth - 1 - coord_cur[IY];
          sign_v = -1.0;
        }
        if (dir == DIR_Y and face == FACE_RIGHT) {
          coord_in[IY] = 2 * by + 2 * ghostWidth - 1 - coord_cur[IY];
          sign_v = -1.0;
        }
      } // end REFLECTING

      // index from which data will be copied
      // convert coord_in into an index

      // version 1 : use global array U
      // uint32_t index = (coord_in[IX]-ghostWidth) + bx*(coord_in[IY]-ghostWidth);
      // Ugroup(index_cur, fm[ID], iOct_local) = U(index, fm[ID], iOct);
      // Ugroup(index_cur, fm[IP], iOct_local) = U(index, fm[IP], iOct);
      // Ugroup(index_cur, fm[IU], iOct_local) = U(index, fm[IU], iOct) * sign_u;
      // Ugroup(index_cur, fm[IV], iOct_local) = U(index, fm[IV], iOct) * sign_v;

      // version 2 : use array Ugroup and
      // assume inner cells have already been copied 
      uint32_t index = coord_in[IX] + (bx+2*ghostWidth)*coord_in[IY];
      Ugroup(index_cur, fm[ID], iOct_local) = Ugroup(index, fm[ID], iOct_local);
      Ugroup(index_cur, fm[IP], iOct_local) = Ugroup(index, fm[IP], iOct_local);
      Ugroup(index_cur, fm[IU], iOct_local) = Ugroup(index, fm[IU], iOct_local) * sign_u;
      Ugroup(index_cur, fm[IV], iOct_local) = Ugroup(index, fm[IV], iOct_local) * sign_v;

      if (copy_gravity) { // Gravity at bc is not well defined, is it ?
        for (int dim=0; dim < ndim; ++dim)
          Ggroup(index_cur, dim, iOct_local) = Ggroup(index, dim, iOct_local);
      } // end if copy
      
    } // end if admissible values

  } // fill_ghost_face_2d_external_border

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct) from
   * a neighbor octant in case neighbor has the same size (i.e.
   * same AMR level).
   *
   * \param[in] iOct global index to current octant
   * \param[in] iOct_local local index (i.e. inside group) to current octant
   * \param[in] iOct_neigh global index to neighbor octant
   * \param[in] is_ghost boolean value, true if neighbor is MPI ghost octant
   * \param[in] index integer used to map the ghost cell to fill
   * \param[in] dir identifies direction of the face border to be filled
   * \param[in] face are we dealing with a left or right interface (as seen from current cell)
   * 
   * Remember that a left interface (for current octant) is a right interface for neighbor octant.
   */
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

    const index_t size_borderX = ghostWidth*by;
    const index_t size_borderY = bx*ghostWidth; 
    
    // make sure index is valid, i.e. inside the range of admissible values
    if ( (index < size_borderX and dir == DIR_X) or
         (index < size_borderY and dir == DIR_Y) ) {

      // in case of a X border
      // border sizes for input  cell are : ghostWidth,by
      // border sizes for output cell are : ghostWidth,by+2*ghostWidth

      // in case of a Y border
      // border sizes for input  cell are : bx             ,ghostWidth
      // border sizes for output cell are : bx+2*ghostWidth,ghostWidth

      // compute cell coordinates inside border
      coord_t coord_border = dir == DIR_X ? 
        index_to_coord(index, ghostWidth, by) :
        index_to_coord(index, bx, ghostWidth) ;
      
      // compute cell coordinates inside ghosted block of the receiving octant (current)
      coord_t coord_cur = {coord_border[IX],
                           coord_border[IY], 
                           0};

      // shift coord to face center
      if (dir == DIR_X)
        coord_cur[IY] += ghostWidth;
      if (dir == DIR_Y)
        coord_cur[IX] += ghostWidth;

      if ( face == FACE_RIGHT ) {
        // if necessary shift coordinate to the right
        if (dir == DIR_X)
          coord_cur[IX] += (bx+ghostWidth);
        if (dir == DIR_Y)
          coord_cur[IY] += (by+ghostWidth);
      }
      
      // compute corresponding index in the ghosted block (current octant) 
      uint32_t index_cur = coord_cur[IX] + (bx+2*ghostWidth)*coord_cur[IY];

      // if necessary, shift border coords to access input (neighbor octant) cell data
      if ( face == FACE_LEFT ) {
        if ( dir == DIR_X )
          coord_border[IX] += (bx - ghostWidth);
        if ( dir == DIR_Y )
          coord_border[IY] += (by - ghostWidth);
      }

      uint32_t index_border = coord_border[IX] + bx * coord_border[IY];

      if (is_ghost) {
        Ugroup(index_cur, fm[ID], iOct_local) = U_ghost(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U_ghost(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U_ghost(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U_ghost(index_border, fm[IV], iOct_neigh);

        if (copy_gravity) { 
          for (int dim=0; dim < ndim; ++dim)
            Ggroup(index_cur, dim, iOct_local) = Gravity_ghost(index_border, dim, iOct_neigh);
        }
      } else {
        Ugroup(index_cur, fm[ID], iOct_local) = U(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U(index_border, fm[IV], iOct_neigh);

        if (copy_gravity)
          for (int dim=0; dim < ndim; ++dim)
            Ggroup(index_cur, dim, iOct_local) = Gravity(index_border, dim, iOct_neigh);
      }

    } // end if admissible values for index

  } // fill_ghost_face_2d_same_size

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct) from
   * a neighbor octant in case neighbor is LARGER (i.e.
   * one level less than current octant's level).
   *
   * \param[in] iOct global index of current octant
   * \param[in] iOct_local local index (i.e. inside group) of current octant
   * \param[in] iOct_neigh global index of neighbor octant
   * \param[in] is_ghost boolean value, true if neighbor is MPI ghost octant
   * \param[in] index integer used to map the ghost cell to fill
   * \param[in] dir identifies direction of the face border to be filled
   * \param[in] face are we dealing with a left or right interface (as seen from current cell)
   * \param[in] loc identifies relative location of current octant and its larger neighbor
   * 
   * Remember that a left interface (for current octant) is a right interface for neighbor octant.
   *
   * Difficulty is to deal with these two possibilities:
   * - current  (small) octant on the right
   * - neighbor (large) octant on the left
   *
   * These 2 situations schematically are
   *  ______               ______    __
   * |      |             |      |  X  |
   * |      |   __    or  |      |  X__|
   * |      |  X  |       |      |
   * |______|  X__|       |______|
   *
   * In this function, we want to fill the "X" ghost cells using data from
   * the (larger) neighbor octant.
   *
   * TODO : do something about hanging nodes, and filling corner ghost cells...
   */
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_face_2d_larger_size(uint32_t iOct,
                                      uint32_t iOct_local,
                                      uint32_t iOct_neigh,
                                      bool     is_ghost,
                                      index_t  index,
                                      DIR_ID   dir,
                                      FACE_ID  face,
                                      NEIGH_LOC loc) const
  {
    
    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    const index_t size_borderX = ghostWidth*by;
    const index_t size_borderY = bx*ghostWidth;

    // make sure index is valid, i.e. inside the range of admissible values
    if ((index < size_borderX and dir == DIR_X) or
        (index < size_borderY and dir == DIR_Y)) {

      // compute cell coordinates inside border
      coord_t coord_border = dir == DIR_X ? 
        index_to_coord(index, ghostWidth, by) :
        index_to_coord(index, bx, ghostWidth) ;
      
      // compute cell coordinates inside ghosted block of the receiving octant (current)
      coord_t coord_cur = {coord_border[IX],
                           coord_border[IY], 
                           0};

      // shift coord to face center
      if (dir == DIR_X)
        coord_cur[IY] += ghostWidth;
      if (dir == DIR_Y)
        coord_cur[IX] += ghostWidth;

      if ( face == FACE_RIGHT ) {
        // if necessary shift coordinate to the right
        if (dir == DIR_X)
          coord_cur[IX] += (bx+ghostWidth);
        if (dir == DIR_Y)
          coord_cur[IY] += (by+ghostWidth);
      }
      
      // compute corresponding index in the ghosted block (current octant) 
      uint32_t index_cur = coord_cur[IX] + (bx+2*ghostWidth)*coord_cur[IY];

      // if necessary, shift border coords to access input (neighbor octant) cell data
      if ( face == FACE_LEFT ) {
        if ( dir == DIR_X )
          coord_border[IX] = (coord_border[IX] + 2*bx - ghostWidth);
        if ( dir == DIR_Y )
          coord_border[IY] = (coord_border[IY] + 2*by - ghostWidth);
      }

      // remap to take into account that neighbor is actually larger
      coord_border[IX] /= 2;
      coord_border[IY] /= 2;

      if (loc == NEIGH_POS_1 and dir == DIR_X)
        coord_border[IY] += by/2;

      if (loc == NEIGH_POS_1 and dir == DIR_Y)
        coord_border[IX] += bx/2;

      uint32_t index_border = coord_border[IX] + bx * coord_border[IY];

      if (is_ghost) {
        Ugroup(index_cur, fm[ID], iOct_local) = U_ghost(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U_ghost(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U_ghost(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U_ghost(index_border, fm[IV], iOct_neigh);

        if (copy_gravity) {
          for (int dim=0; dim < ndim; ++dim)
            Ggroup(index_cur, dim, iOct_local) = Gravity_ghost(index_border, dim, iOct_neigh);
        }
      } else {
        Ugroup(index_cur, fm[ID], iOct_local) = U(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U(index_border, fm[IV], iOct_neigh);

        if (copy_gravity) {
          for (int dim=0; dim < ndim; ++dim)
            Ggroup(index_cur, dim, iOct_local) = Gravity(index_border, dim, iOct_neigh);
        }
      }

    } // end if admissible values for index

  } // fill_ghost_face_2d_larger_size

  // ==============================================================
  // ==============================================================
  /**
   * Fill (copy) ghost cell data of current octant (iOct) from
   * a neighbor octant in case neighbor is SMALLER (i.e.
   * one level more than current octant's level).
   *
   * \param[in] iOct global index to current octant
   * \param[in] iOct_local local index (i.e. inside group) to current octant
   * \param[in] iOct_neigh array of size 2 of global indexes to neighbor octants
   * \param[in] is_ghost array of size 2 of boolean values, true if neighbor is MPI ghost octant
   * \param[in] index integer used to map the ghost cell to fill
   * \param[in] dir identifies direction of the face border to be filled
   * \param[in] face are we dealing with a left or right interface (as seen from current cell)
   * \param[in] loc array of size 2 which identifies relative location of current octant and its smaller neighbors
   * 
   * Remember that a left interface (for current octant) is a right interface for neighbor octant.
   *
   * Difficulty is to deal with these two possibilities:
   * current  (large) octant on the right
   * neighbor (small) octant on the left
   *
   * These 2 situations schematically are :
   *        _______             __      _______ 
   *       |       |           |  |    X       |
   *  __   |       |      or   |__|    X       |
   * |  |  X       |                   |       |
   * |__|  x_______|                   |_______|
   *
   * NEIGH_POS_0               NEIGH_POS_1
   *
   *
   */
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_face_2d_smaller_size(uint32_t iOct,
                                       uint32_t iOct_local,
                                       uint32_t iOct_neigh[2],
                                       bool     is_ghost[2],
                                       index_t  index,
                                       DIR_ID   dir,
                                       FACE_ID  face,
                                       NEIGH_LOC loc[2]) const
  {
    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    // hydrodynamics state variables (initialized to zero)
    // used to accumulate values from smaller octant into
    // large octant ghost cells
    HydroState2d q = {0, 0, 0, 0};
    real_t gravity[ndim]{0.0};

    const index_t size_borderX = ghostWidth*by;
    const index_t size_borderY = bx*ghostWidth;

    // make sure index is valid, 
    // i.e. inside the range of admissible values
    if ((index < size_borderX and dir == DIR_X) or
        (index < size_borderY and dir == DIR_Y)) {

      // compute cell coordinates inside border of current block,
      // ghost width not taken into account now
      coord_t coord_cur = dir == DIR_X ? 
        index_to_coord(index, ghostWidth, by) :
        index_to_coord(index, bx, ghostWidth) ;

      // initialize cell coordinates inside neighbor block 
      // of the receiving octant (i.e. current octant)
      coord_t coord_border = {coord_cur[IX],
                              coord_cur[IY], 
                              0};

      // precisely compute coord_cur
      // shift coord to face center
      if (dir == DIR_X)
        coord_cur[IY] += ghostWidth;
      if (dir == DIR_Y)
        coord_cur[IX] += ghostWidth;

      // make sure coord_cur is actually mapping the right ghost border
      if ( face == FACE_RIGHT ) {
        // if necessary shift coordinate to the right
        if (dir == DIR_X)
          coord_cur[IX] += (bx+ghostWidth);
        if (dir == DIR_Y)
          coord_cur[IY] += (by+ghostWidth);
      }

      
      // compute corresponding index in the ghosted block
      // i.e. current octant
      uint32_t index_cur = coord_cur[IX] + (bx+2*ghostWidth)*coord_cur[IY];

      // loop inside neighbor block (smaller octant) to accumulate values
      for (int8_t iy = 0; iy < 2; ++iy) {
        for (int8_t ix = 0; ix < 2; ++ix) {

          int32_t ii = 2 * coord_border[IX] + ix;
          int32_t jj = 2 * coord_border[IY] + iy;
          
          // select from which neighbor we will take data
          // this should be ok, even if bx/by are odd integers
          uint8_t iNeigh=0;
          if (dir == DIR_X and jj>=by) {
            iNeigh=1;
            jj -= by;
          }
          if (dir == DIR_Y and ii>=bx) {
            iNeigh=1;
            ii -= bx;
          }

          // if necessary, shift border coords to access input data in 
          // neighbor octant
          // remember that : left interface for current octant
          // translates into a right interface for neighbor octants
          if ( face == FACE_LEFT ) {
            if ( dir == DIR_X )
              ii += (bx - 2*ghostWidth);
            if ( dir == DIR_Y )
              jj += (by - 2*ghostWidth);
          }

          // hopefully neighbor octant, cell coordinates are ok now
          // so compute index to access data
          uint32_t index_border = ii + bx * jj;
          
          if (is_ghost[iNeigh]) {
            q[ID] += U_ghost(index_border, fm[ID], iOct_neigh[iNeigh]);
            q[IP] += U_ghost(index_border, fm[IP], iOct_neigh[iNeigh]);
            q[IU] += U_ghost(index_border, fm[IU], iOct_neigh[iNeigh]);
            q[IV] += U_ghost(index_border, fm[IV], iOct_neigh[iNeigh]);

            if (copy_gravity) {
              for (int dim=0; dim < ndim; ++dim)
                gravity[dim] += Gravity_ghost(index_border, dim, iOct_neigh[iNeigh]);
            }
          } else {
            q[ID] += U(index_border, fm[ID], iOct_neigh[iNeigh]);
            q[IP] += U(index_border, fm[IP], iOct_neigh[iNeigh]);
            q[IU] += U(index_border, fm[IU], iOct_neigh[iNeigh]);
            q[IV] += U(index_border, fm[IV], iOct_neigh[iNeigh]);

            if (copy_gravity) {
              for (int dim=0; dim < ndim; ++dim)
                gravity[dim] += Gravity(index_border, dim, iOct_neigh[iNeigh]);
            }
          }

        } // end for ix
      } // end for iy

      // copy back accumulated results
      Ugroup(index_cur, fm[ID], iOct_local) = q[ID]/4;
      Ugroup(index_cur, fm[IP], iOct_local) = q[IP]/4;
      Ugroup(index_cur, fm[IU], iOct_local) = q[IU]/4;
      Ugroup(index_cur, fm[IV], iOct_local) = q[IV]/4;

      if (copy_gravity) {
        for (int dim=0; dim < ndim; ++dim)
          Ggroup(index_cur, dim, iOct_local) = gravity[dim];
      } // end if copy_gravity
    } // end if admissible values for index

  } // fill_ghost_face_2d_smaller_size

  // ==============================================================
  // ==============================================================
  /**
   * this routine is mainly a driver to safely call these three:
   * - fill_ghost_face_2d_same_size
   * - fill_ghost_face_2d_larger_size
   * - fill_ghost_face_2d_smaller_size
   */
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_face_2d(uint32_t iOct, 
                          uint32_t iOct_local, 
                          index_t  index_in, 
                          DIR_ID   dir, 
                          FACE_ID  face) const
  {

    // probe mesh neighbor information
    uint8_t iface = face + 2*dir;
    uint8_t codim = 1;

    // list of neighbors octant id, neighbor through a given face
    std::vector<uint32_t> neigh;
    std::vector<bool> isghost;

    // ask PABLO to find neighbor octant id accross a given face
    // this fill fill vector neigh and isghost
    pmesh->findNeighbours(iOct, iface, codim, neigh, isghost);

    /*
     * first deal with external border
     */
    if (neigh.size() == 0) {

      // if (index_in==0)
      //   printf("[neigh is external] iOct_global=%d iOct_local=%2d ---- dir=%d face=%d \n",iOct, iOct_local, dir, face);

      fill_ghost_face_2d_external_border(iOct, iOct_local, index_in, dir, face);
    } 
    
    /*
     * there one neighbor accross face , either same size or larger
     */
    else if (neigh.size() == 1) {

      // retrieve neighbor octant id
      uint32_t iOct_neigh = neigh[0];

      /* check if neighbor is larger, it means one of the corner of current octant
       * is a hanging node, there are 2 distinct geometrical possibilities, 
       * here illustrated with
       * neighbor (large) octant on the left and current (small) octant on the right 
       *
       *  _______              ______    __
       * |      |             |      |  |  |
       * |      |   __    or  |      |  |__|
       * |      |  |  |       |      |
       * |______|  |__|       |______|  
       */
      if ( pmesh->getLevel(iOct) > pmesh->getLevel(iOct_neigh) ) {

        // if (index_in==0)
        //   printf("[neigh is larger] iOct_global=%d iOct_local=%2d iOct_neigh=%2d ---- \n",iOct, iOct_local, iOct_neigh);

        // Setting interface flag to "bigger"
	Interface_flags(iOct_local) |= (1 << (iface + 6));
	
        NEIGH_LOC loc = get_relative_position_2d(iOct, iOct_neigh, isghost[0], dir, face, NEIGH_IS_LARGER);

        fill_ghost_face_2d_larger_size(iOct, iOct_local, iOct_neigh, isghost[0], index_in, dir, face, loc);

      } else {
        // if (index_in==0)
        //   printf("[neigh has same size] iOct_global=%d iOct_local=%2d iOct_neigh=%2d \n",iOct, iOct_local, iOct_neigh);

        fill_ghost_face_2d_same_size(iOct, iOct_local, iOct_neigh, isghost[0], index_in, dir, face);

      } // end iOct and iOct_neigh have same size

    } // end neigh.size() == 1

    /*
     * there are 2 neighbors accross face, smaller than current octant
     * here we average values read from the smaller octant before copying
     * into ghost cells of the larger octant.
     */
    else if (neigh.size() == 2) {

      // Setting interface flag to "smaller"
      Interface_flags(iOct_local) |= (1<<iface);

      // if (index_in==0)
      //   printf("[neigh has smaller size] iOct_global=%d iOct_local=%2d iOct_neigh0=%2d iOct_neigh1=%2d -- dir=%d face=%d\n",
      //          iOct, iOct_local, neigh[0], neigh[1],dir,face);

      // compute relative position of smaller octant versus current one
      // we do that because I don't know if PABLO always returns the neighbor
      // in that order. To be investigated, it is not clearly stated in PABLO
      // doc (doxygen), it probably require a good reading of PABLO source code
      // and reverse engineering to be sure. Maybe ask PABLO authors in a github issue ?
      NEIGH_LOC loc[2];
      uint32_t iOct_neigh[2] = {neigh[0], neigh[1]};
      bool isghost_neigh[2] = {isghost[0], isghost[1]};

      loc[0] = get_relative_position_2d (iOct, neigh[0], isghost[0], dir, face, NEIGH_IS_SMALLER);
      loc[1] = get_relative_position_2d (iOct, neigh[1], isghost[1], dir, face, NEIGH_IS_SMALLER);

      fill_ghost_face_2d_smaller_size(iOct, iOct_local, iOct_neigh, 
                                      isghost_neigh, index_in, 
                                      dir, face, loc);
      
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
      Interface_flags(iOct_g) = INTERFACE_NONE;

      // perform "vectorized" loop inside a given block data
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, nbCells),
          KOKKOS_LAMBDA(const index_t index) {
            // compute face X,left
            fill_ghost_face_2d(iOct, iOct_g, index, DIR_X, FACE_LEFT);

            // compute face X,right
            fill_ghost_face_2d(iOct, iOct_g, index, DIR_X, FACE_RIGHT);

            // compute face Y,left
            fill_ghost_face_2d(iOct, iOct_g, index, DIR_Y, FACE_LEFT);

            // compute face Y,right
            fill_ghost_face_2d(iOct, iOct_g, index, DIR_Y, FACE_RIGHT);
            
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

  //! heavy data - input - global array for gravity
  DataArrayBlock Gravity;

  //! heavy data - input - ghost array for gravity
  DataArrayBlock Gravity_ghost;

  //! heavy data - input - current ghosted block group for gravity
  DataArrayBlock Ggroup;

  //! id of group of octants to be copied
  uint32_t iGroup;

  //! 2:1 flagging mechanism
  FlagArrayBlock Interface_flags;

  // should we copy gravity ?
  bool copy_gravity;

  // number of dimensions for gravity
  int ndim;

}; // CopyFaceBlockCellDataFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // COPY_FACE_BLOCK_CELL_DATA_FUNCTOR_H_
