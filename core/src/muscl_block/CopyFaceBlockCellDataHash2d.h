/**
 * \file CopyFaceBlockCellDataHash.h
 * \author Pierre Kestener
 */
#ifndef COPY_FACE_BLOCK_CELL_DATA_HASH_FUNCTOR_2D_H_
#define COPY_FACE_BLOCK_CELL_DATA_HASH_FUNCTOR_2D_H_

#include "shared/FieldManager.h"
#include "shared/kokkos_shared.h"

// utils hydro
#include "shared/utils_hydro.h"

// utils block for :
// - enum NEIGH_LOC / NEIGH_SIZE
#include "muscl_block/utils_block.h"

#include "shared/AMRMetaData.h"

namespace dyablo
{
namespace muscl_block
{

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
 * This new version of CopyFaceBlockCellDataFunctor uses AMRMetaData
 * instead of bitpit mesh (not kokkos compatible).
 *
 * \tparam dim dimension is 2 or 3, necessary for AMRMetaData
 */
template<>
class CopyFaceBlockCellDataHashFunctor<2>
{

private:
  uint32_t nbTeams; //!< number of thread teams
  
public:
  using index_t = int32_t;
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<index_t>>;
  using thread_t = team_policy_t::member_type;

  // AMR related type alias
  static constexpr int dim = 2;
  using key_t = typename AMRMetaData<dim>::key_t;
  using value_t = typename AMRMetaData<dim>::value_t;

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  using NEIGH_LEVEL    = typename AMRMetaData<dim>::NEIGH_LEVEL;
  using NEIGH_POSITION = typename AMRMetaData<dim>::NEIGH_POSITION;

  /**
   * enum used to identify the relative position of an octant
   * versus a face neighbor octant of larger or smaller size.
   *
   * In 2d, only 2 possibilities
   * In 3d, there are 4 possibilities
   */
  // enum NEIGH_LOC : uint8_t
  // {
  //   NEIGH_POS_0 = 0,
  //   NEIGH_POS_1 = 1,
  //   NEIGH_POS_2 = 2,
  //   NEIGH_POS_3 = 3
  // };

  // enum NEIGH_SIZE : uint8_t
  // {
  //   NEIGH_IS_SMALLER   = 0,
  //   NEIGH_IS_LARGER    = 1,
  //   NEIGH_IS_SAME_SIZE = 2
  // };

  /**
   *
   * \param[in] mesh reference a AMRMetaData object for connectivity 
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
  CopyFaceBlockCellDataHashFunctor(AMRMetaData<dim> mesh,
                                   HydroParams params,
                                   id2index_t fm,
                                   blockSize_t blockSizes, 
                                   uint32_t ghostWidth,
                                   uint32_t nbOctsPerGroup,
                                   DataArrayBlock U,
                                   DataArrayBlock U_ghost,
                                   DataArrayBlock Ugroup,
                                   uint32_t iGroup,
                                   FlagArrayBlock Interface_flags) :
    mesh(mesh), nbOcts(mesh.nbOctants()), 
    params(params),
    fm(fm), blockSizes(blockSizes), 
    ghostWidth(ghostWidth),
    nbOctsPerGroup(nbOctsPerGroup), 
    U(U), 
    U_ghost(U_ghost),
    Ugroup(Ugroup), 
    iGroup(iGroup),
    Interface_flags(Interface_flags)
  {

    // in 2d, bz and bz_g are not used

    bx_g = blockSizes[IX] + 2 * ghostWidth;
    by_g = blockSizes[IY] + 2 * ghostWidth;
    bz_g = blockSizes[IZ] + 2 * ghostWidth;

  };

  // static method which does it all: create and execute functor
  static void apply(AMRMetaData<dim> mesh,
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

    CopyFaceBlockCellDataHashFunctor functor(mesh, params, fm, 
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
    Kokkos::parallel_for("dyablo::muscl_block::CopyFaceBlockCellDataHashFunctor",
                         policy, functor);
  }
  
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
        (index_in < size_borderY and dir == DIR_Y)) 
    {

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

      if ( face == FACE_RIGHT )
      {
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
         (index < size_borderY and dir == DIR_Y) )
    {

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

      if ( face == FACE_RIGHT )
      {
        // if necessary shift coordinate to the right
        if (dir == DIR_X)
          coord_cur[IX] += (bx+ghostWidth);
        if (dir == DIR_Y)
          coord_cur[IY] += (by+ghostWidth);
      }
      
      // compute corresponding index in the ghosted block (current octant) 
      uint32_t index_cur = coord_cur[IX] + (bx+2*ghostWidth)*coord_cur[IY];

      // if necessary, shift border coords to access input (neighbor octant) cell data
      if ( face == FACE_LEFT )
      {
        if ( dir == DIR_X )
          coord_border[IX] += (bx - ghostWidth);
        if ( dir == DIR_Y )
          coord_border[IY] += (by - ghostWidth);
      }

      uint32_t index_border = coord_border[IX] + bx * coord_border[IY];

      if (is_ghost)
      {
        Ugroup(index_cur, fm[ID], iOct_local) = U_ghost(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U_ghost(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U_ghost(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U_ghost(index_border, fm[IV], iOct_neigh);
      }
      else
      {
        Ugroup(index_cur, fm[ID], iOct_local) = U(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U(index_border, fm[IV], iOct_neigh);
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
                                      NEIGH_POSITION loc) const
  {
    
    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    const index_t size_borderX = ghostWidth*by;
    const index_t size_borderY = bx*ghostWidth;

    // make sure index is valid, i.e. inside the range of admissible values
    if ((index < size_borderX and dir == DIR_X) or
        (index < size_borderY and dir == DIR_Y))
    {

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

      if ( face == FACE_RIGHT )
      {
        // if necessary shift coordinate to the right
        if (dir == DIR_X)
          coord_cur[IX] += (bx+ghostWidth);
        if (dir == DIR_Y)
          coord_cur[IY] += (by+ghostWidth);
      }
      
      // compute corresponding index in the ghosted block (current octant) 
      uint32_t index_cur = coord_cur[IX] + (bx+2*ghostWidth)*coord_cur[IY];

      // if necessary, shift border coords to access input (neighbor octant) cell data
      if ( face == FACE_LEFT )
      {
        if ( dir == DIR_X )
          coord_border[IX] = (coord_border[IX] + 2*bx - ghostWidth);
        if ( dir == DIR_Y )
          coord_border[IY] = (coord_border[IY] + 2*by - ghostWidth);
      }

      // remap to take into account that neighbor is actually larger
      coord_border[IX] /= 2;
      coord_border[IY] /= 2;

      if (loc == NEIGH_POSITION::NEIGH_POS_1 and dir == DIR_X)
        coord_border[IY] += by/2;

      if (loc == NEIGH_POSITION::NEIGH_POS_1 and dir == DIR_Y)
        coord_border[IX] += bx/2;

      uint32_t index_border = coord_border[IX] + bx * coord_border[IY];

      if (is_ghost)
      {
        Ugroup(index_cur, fm[ID], iOct_local) = U_ghost(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U_ghost(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U_ghost(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U_ghost(index_border, fm[IV], iOct_neigh);
      }
      else
      {
        Ugroup(index_cur, fm[ID], iOct_local) = U(index_border, fm[ID], iOct_neigh);
        Ugroup(index_cur, fm[IP], iOct_local) = U(index_border, fm[IP], iOct_neigh);
        Ugroup(index_cur, fm[IU], iOct_local) = U(index_border, fm[IU], iOct_neigh);
        Ugroup(index_cur, fm[IV], iOct_local) = U(index_border, fm[IV], iOct_neigh);
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
                                       FACE_ID  face) const
  {
    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    // hydrodynamics state variables (initialized to zero)
    // used to accumulate values from smaller octant into
    // large octant ghost cells
    HydroState2d q = {0, 0, 0, 0};

    const index_t size_borderX = ghostWidth*by;
    const index_t size_borderY = bx*ghostWidth;

    // make sure index is valid, 
    // i.e. inside the range of admissible values
    if ((index < size_borderX and dir == DIR_X) or
        (index < size_borderY and dir == DIR_Y))
    {

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
      if ( face == FACE_RIGHT )
      {
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
      for (int8_t iy = 0; iy < 2; ++iy)
      {
        for (int8_t ix = 0; ix < 2; ++ix)
        {

          int32_t ii = 2 * coord_border[IX] + ix;
          int32_t jj = 2 * coord_border[IY] + iy;
          
          // select from which neighbor we will take data
          // this should be ok, even if bx/by are odd integers
          uint8_t iNeigh=0;
          if (dir == DIR_X and jj>=by)
          {
            iNeigh=1;
            jj -= by;
          }
          
          if (dir == DIR_Y and ii>=bx)
          {
            iNeigh=1;
            ii -= bx;
          }

          // if necessary, shift border coords to access input data in 
          // neighbor octant
          // remember that : left interface for current octant
          // translates into a right interface for neighbor octants
          if ( face == FACE_LEFT )
          {
            if ( dir == DIR_X )
              ii += (bx - 2*ghostWidth);
            if ( dir == DIR_Y )
              jj += (by - 2*ghostWidth);
          }

          // hopefully neighbor octant, cell coordinates are ok now
          // so compute index to access data
          uint32_t index_border = ii + bx * jj;
          
          if (is_ghost[iNeigh]) 
          {
            q[ID] += U_ghost(index_border, fm[ID], iOct_neigh[iNeigh]);
            q[IP] += U_ghost(index_border, fm[IP], iOct_neigh[iNeigh]);
            q[IU] += U_ghost(index_border, fm[IU], iOct_neigh[iNeigh]);
            q[IV] += U_ghost(index_border, fm[IV], iOct_neigh[iNeigh]);
          } 
          else
          {
            q[ID] += U(index_border, fm[ID], iOct_neigh[iNeigh]);
            q[IP] += U(index_border, fm[IP], iOct_neigh[iNeigh]);
            q[IU] += U(index_border, fm[IU], iOct_neigh[iNeigh]);
            q[IV] += U(index_border, fm[IV], iOct_neigh[iNeigh]);
          }

        } // end for ix
      } // end for iy

      // copy back accumulated results
      Ugroup(index_cur, fm[ID], iOct_local) = q[ID]/4;
      Ugroup(index_cur, fm[IP], iOct_local) = q[IP]/4;
      Ugroup(index_cur, fm[IU], iOct_local) = q[IU]/4;
      Ugroup(index_cur, fm[IV], iOct_local) = q[IV]/4;
    } // end if admissible values for index

  } // fill_ghost_face_2d_smaller_size

  // ==============================================================
  // ==============================================================
  /**
   * this routine is mainly a driver to safely call these three:
   * - fill_ghost_face_2d_same_size
   * - fill_ghost_face_2d_larger_size
   * - fill_ghost_face_2d_smaller_size
   *
   * \param[in] iOct octant id (regular) among all octant in current MPI process
   * \param[in] iOct_local octant id inside current group/batch of octants processed
   * \param[in] morton index of current octant
   * \param[in] level of current octant
   * \param[in] index_in
   * \param[in] dir is direction of the face
   * \param[in] face is 0 (left) or right (1)
   * \param[in] status is a small integer encode neighbor level difference relative to current octant
   */
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_face_2d(uint32_t iOct, 
                          uint32_t iOct_local,
                          uint64_t morton,
                          uint8_t  level,
                          index_t  index_in, 
                          DIR_ID   dir, 
                          FACE_ID  face,
                          AMRMetaData<2>::neigh_level_status_t status_level,
                          AMRMetaData<2>::neigh_rel_pos_status_t status_rel_pos) const
  {

    // probe mesh neighbor information
    uint8_t iface = face + 2*dir;

    // decode neigh level status (larger ? same size ? smaller ?)
    using NEIGH_LEVEL = AMRMetaData<2>::NEIGH_LEVEL;
    NEIGH_LEVEL nl = static_cast<NEIGH_LEVEL>( (status_level >> (2*iface)) & 0x3 );

    // get hashmap from mesh, we'll need it to retrieve neighbor octant id
    auto hashmap = mesh.hashmap();

    // total number of octants
    //uint32_t nbOcts = mesh.nbOctants();

    /*
     * first deal with external border
     */
    if (nl == NEIGH_LEVEL::NEIGH_IS_EXTERNAL_BORDER)
    {

      fill_ghost_face_2d_external_border(iOct, iOct_local, index_in, dir, face);

    } 

    /*
     * there is one neighbor, larger at level-1
     * we further need the relative position to precisely compute it morton key.
     */
    else if (nl == NEIGH_LEVEL::NEIGH_IS_LARGER)
    {

      /*
       * retrieve neighbor octant id through the given face id
       */
      
      /* neighbor is larger, there are 2 distinct geometrical possibilities, 
       * here illustrated with
       * neighbor (large) octant on the left and current (small) octant on the right 
       *
       *  ______               ______    __
       * |N     |             |N     |  |C |
       * |      |   __    or  |      |  |__|
       * |      |  |C |       |      |
       * |______|  |__|       |______|  
       */
      // decode neigh relative position
      NEIGH_POSITION rel_pos = 
        static_cast<NEIGH_POSITION>( (status_rel_pos >> (iface)) & 0x1 );

      // neighbor hash key
      key_t key_n;

      key_n[0] = get_neighbor_morton(morton, level, level-1, iface, rel_pos);
      key_n[1] = level-1;

      // hashmap index - index_n has to be a valid index,
      // or else we're in deep trouble
      auto index_n = hashmap.find(key_n);
      
      //if ( invalid_index != index_n )
      //{
      uint32_t iOct_neigh = hashmap.value_at(index_n);
      //}
            
      // Setting interface flag to "bigger" -- PROBABLY NOT NEEDED ANYMORE
      Interface_flags(iOct_local) |= (1 << (iface + 6));
	
      bool isGhost = iOct_neigh>=nbOcts;
      
      fill_ghost_face_2d_larger_size(iOct, iOct_local, iOct_neigh, isGhost, index_in, dir, face, rel_pos);
      
    } // NEIGH_IS_LARGER

    else if (nl == NEIGH_LEVEL::NEIGH_IS_SAME_SIZE)
    {

      // neighbor hash key
      key_t key_n;

      key_n[0] = get_neighbor_morton(morton, level, iface);
      key_n[1] = level;

      // hashmap index - index_n has to be a valid index,
      // or else we're in deep trouble
      auto index_n = hashmap.find(key_n);
      
      uint32_t iOct_neigh = hashmap.value_at(index_n);

      bool isGhost = iOct_neigh>=nbOcts;

      fill_ghost_face_2d_same_size(iOct, iOct_local, iOct_neigh, isGhost, index_in, dir, face);

    } // end NEIGH_IS_SAME_SIZE

    /*
     * there are 2 neighbors across face, smaller than current octant
     * here we average values read from the smaller octant before copying
     * into ghost cells of the larger octant.
     */
    else if (nl == NEIGH_LEVEL::NEIGH_IS_SMALLER)
    {

      // Setting interface flag to "smaller" -- PROBABLY NOT NEEDED ANYMORE
      Interface_flags(iOct_local) |= (1<<iface);

      // retrieve the 2 neighbor octant ids
      uint32_t iOct_neigh[2];
      bool isghost_neigh[2];

      for (int ineigh=0; ineigh<2; ++ineigh)
      {

        // neighbor hash key
        key_t key_n;

        key_n[0] = get_neighbor_morton(morton, level, level+1, iface, ineigh);
        key_n[1] = level+1;

        // hashmap index - index_n has to be a valid index,
        // or else we're in deep trouble
        auto index_n = hashmap.find(key_n);
        
        iOct_neigh[ineigh] = hashmap.value_at(index_n);
        isghost_neigh[ineigh] = iOct_neigh[ineigh]>= nbOcts;

      }

      fill_ghost_face_2d_smaller_size(iOct, iOct_local, iOct_neigh, 
                                      isghost_neigh, index_in, 
                                      dir, face);
      
    } // end NEIGH_IS_SMALLER

  } // fill_ghost_face_2d

  // ==============================================================
  // ================================================
  // ================================
  // 2D version.
  // ================================
  // ================================================
  // ==============================================================
  //! functor for 2d 
  // template<int dim_ = dim>
  // KOKKOS_INLINE_FUNCTION
  // void operator()(typename std::enable_if<dim_==2, team_policy_t::member_type>::type member) const 
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const 
  {

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_g = member.league_rank();

    // total number of octants
    //uint32_t nbOcts = mesh.nbOctants();

    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    // maximun number of ghost cells to fill (per face)
    // this number will be used to set the number of working threads
    uint32_t bmax = bx < by ? by : bx;
    uint32_t nbCells = bmax*ghostWidth;

    // get a const ref on the array of neighbor "level status"
    const auto neigh_level_status = mesh.neigh_level_status_array();

    // get a const ref on the array of neighbor "relative position status"
    const auto neigh_rel_pos_status = mesh.neigh_rel_pos_status_array();

    // get a const ref on the array of morton keys
    const auto morton_keys = mesh.morton_keys();

    // get a const ref on the array of levels
    const auto levels = mesh.levels();

    //
    // THE main loop
    //
    while (iOct < iOctNextGroup and iOct < nbOcts)
    {
      Interface_flags(iOct_g) = INTERFACE_NONE;

      // get neigh level status
      auto neigh_lv_status = neigh_level_status(iOct);

      // get neigh relative position status
      auto neigh_rp_status = neigh_rel_pos_status(iOct);

      uint64_t morton_key = morton_keys(iOct);
      uint8_t level = levels(iOct);

      // perform "vectorized" loop inside a given block data
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        [&](const index_t index)
        {
          // compute face X,left
          fill_ghost_face_2d(iOct, iOct_g, morton_key, level, index, DIR_X, FACE_LEFT, neigh_lv_status, neigh_rp_status);
          
          // compute face X,right
          fill_ghost_face_2d(iOct, iOct_g, morton_key, level, index, DIR_X, FACE_RIGHT, neigh_lv_status, neigh_rp_status);
          
          // compute face Y,left
          fill_ghost_face_2d(iOct, iOct_g, morton_key, level, index, DIR_Y, FACE_LEFT, neigh_lv_status, neigh_rp_status);
          
          // compute face Y,right
          fill_ghost_face_2d(iOct, iOct_g, morton_key, level, index, DIR_Y, FACE_RIGHT, neigh_lv_status, neigh_rp_status);
          
        }); // end TeamVectorRange
      
      // increase current octant location both in U and Ugroup
      iOct   += nbTeams;
      iOct_g += nbTeams;

    } // end while iOct inside current group of octants

  } // operator() - 2d version

  // ==============================================================
  // ================================================
  // ================================
  // 3D version.
  // ================================
  // ================================================
  // ==============================================================
  //! functor for 3d 
  // template<int dim_ = dim>
  // KOKKOS_INLINE_FUNCTION
  // void operator()(typename std::enable_if<dim_==3, team_policy_t::member_type>::type member) const 
  // {

  //   // do nothing, shouldn't be here....

  // } // operator() - 3d version

  //! AMR mesh
  AMRMetaData<dim> mesh;

  //! number of regular octants
  uint32_t nbOcts;

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

  //! 2:1 flagging mechanism
  FlagArrayBlock Interface_flags;

}; // CopyFaceBlockCellDataHashFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // COPY_FACE_BLOCK_CELL_DATA_HASH_FUNCTOR_3D_H_
