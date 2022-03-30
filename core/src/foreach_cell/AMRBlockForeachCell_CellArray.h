#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "mpi/GhostCommunicator.h"

namespace dyablo {


class AMRBlockForeachCell_Patch;

namespace AMRBlockForeachCell_CellArray_impl{

#define CELLINDEX_INVALID CellIndex{{0,true},0,0,0,0,0,0,CellIndex::INVALID}
#define CELLINDEX_BOUNDARY CellIndex{{0,true},0,0,0,0,0,0,CellIndex::BOUNDARY}

template< typename View_t >
class CellArray_base;

using CellArray_global = CellArray_base<DataArrayBlock>;
class CellArray_global_ghosted;

struct CellIndex
{
  LightOctree::OctantIndex iOct;
  uint32_t i,j,k;
  uint32_t bx,by,bz;
  enum Status {
    LOCAL_TO_BLOCK,
    SAME_SIZE,
    SMALLER,
    BIGGER,
    BOUNDARY,
    INVALID
  } status;

  /**
   * Difference of level between current cell and original cell used in convert_index()
   * note : is 0 for indexes that are local to the block
   **/
  KOKKOS_INLINE_FUNCTION
  int level_diff() const
  {
    return (status==BIGGER)-(status==SMALLER);
  }

  /**
   * Conversion during convert_index() resulted in a CellIndex outside of local block
   * and neighbor search was impossible ( has_neighborhood() was false )
   **/
  KOKKOS_INLINE_FUNCTION
  bool is_valid() const
  {
    return (status!=INVALID) && (status!=BOUNDARY);
  }

  KOKKOS_INLINE_FUNCTION
  bool is_boundary() const
  {
    return status==BOUNDARY;
  }

  /**
   * convert_index() did not need neighbor search to get this index. 
   * If is_local() == false, this index can only be used to interact with 
   * arrays where has_neighborhood() == true (access AND conversion)
   **/
  KOKKOS_INLINE_FUNCTION
  bool is_local() const
  {
    return status==LOCAL_TO_BLOCK;
  }

  /**
   * Get local octant index (cannot be ghost)
   **/
  KOKKOS_INLINE_FUNCTION
  uint32_t getOct() const
  {
    assert(!iOct.isGhost);
    return iOct.iOct;
  }

  using offset_t = Kokkos::Array< int16_t, 3 >;


  /**
   * Compute neighbor index in a ghosted with neighbor-octant search.
   * 
   * @param array a ghosted array compatible with the current CellIndex (same block size)
   * 
   * Offseting outside the block returns a CellIndex pointing to the 
   * corresponding neighbor octant. In this case `is_local()==false` and resulting 
   * cell might be non-conforming (`level_diff()` might be != 0).
   * When level_diff() >= 0, result points to the only neighbor cell
   * When level_diff() < 0 (neighbor is smaller) , result points to the 'lower-left' adjacent neighbor cell 
   * (neighbor with smallest Morton). To get sibling cells, use getNeighbor_ghost({0/1,0/1,0/1}).
   * NOTE: If offset >= 2 outside of the block, resulting cell is one of the subcells in same-size equivalent neighbor,
   * but if cells are of different size, behavior is undefined (therefore, block size must be >= 2*|offset| - 1 )
   *    |--|y-|--|--|--x--|-----| getNeighbor_ghost(x, {-2,0,0}) = y
   * NOTE : If block size is pair and >= 2*offset, smaller siblings are guaranteed to be in the same block, then getNeighbor() can be used
   **/
  KOKKOS_INLINE_FUNCTION
  CellIndex getNeighbor_ghost( const offset_t& offset, const CellArray_global_ghosted& array ) const;

  /**
   * Compute neighbor index with neighbor-octant search.
   * 
   * @param array an array compatible with the current CellIndex (same block size)   * 
   * Offseting outside the block returns an invalid (is_valid()==false) CellIndex.
   **/
  template< typename View_t >
  KOKKOS_INLINE_FUNCTION
  CellIndex getNeighbor_ghost( const offset_t& offset, const CellArray_base<View_t>& array ) const;

  /**
   * Compute neighbor index inside local block
   * Offseting outside the local block is undefined behavior
   **/
  KOKKOS_INLINE_FUNCTION
  CellIndex getNeighbor( const offset_t& offset ) const;

  KOKKOS_INLINE_FUNCTION
  CellIndex operator+( const offset_t& offset ) const
  {
    return getNeighbor(offset);
  }
};


template< typename View_t_ >
class CellArray_base{
public:
  using View_t = View_t_;

  View_t U;    
  uint32_t bx,by,bz;
  uint32_t nbOcts;
  id2index_t fm;

  /**
   * Convert cell index used for another array into an 
   * index compatible with current array. 
   * This method is assuming that the resulting index is valid and in the same block.
   * Neighbor search is never performed, and a resulting index outside of block results in undefined behavior
   **/
  KOKKOS_INLINE_FUNCTION
  CellIndex convert_index(const CellIndex& iCell) const;

  /**
   * Same as convert_index, but returns an index with is_valid() == false when resulting index is outside current block
   **/
  KOKKOS_INLINE_FUNCTION
  CellIndex convert_index_ghost(const CellIndex& iCell) const;

  /**
   *  Get value of field for cell iCell
   * 
   * @param iCell must have a block size compatible with array
   **/
  KOKKOS_INLINE_FUNCTION
  real_t& at( const CellIndex& iCell, VarIndex field ) const;
};

class CellArray_global_ghosted : public CellArray_global{
public :
  View_t Ughost;
  LightOctree lmesh;

  CellArray_global_ghosted() = default;
  CellArray_global_ghosted( const CellArray_global_ghosted& ) = default;

  CellArray_global_ghosted( const CellArray_global& a, const View_t& Ughost, const LightOctree& lmesh )
    : CellArray_global(a), Ughost(Ughost), lmesh(lmesh)
  {}

  void update_lightOctree(  const LightOctree& lmesh ) // TODO remove this once Kokkos arrays are not resized manually anymore.
  {
    this->lmesh = lmesh;
    this->nbOcts = lmesh.getNumOctants();
  }


  /**
   * Convert cell index used for another array into an index compatible with current array. 
   * This may of may not perform a neighbor search for indexes outside of 
   * block depending on how the array was created (see get_global_array, get_global_ghosted_array and allocate_patch_tmp)
   * If a neighbor search is performed the created index has the non-local status (is_local()==false)
   * Converted indexes keep their non-local status after conversion, but not their level difference.
   * Neighbor search is never performed on non-local indexes, is_valid()==false when resulting index is outside of block.
   * All subcells of the ghost cell must be of the same size. (i.e. block size >= 2*offset).
   * NOTE : If it is not the case (i.e. cell-based), use convert_index_getNeighbor() instead;
   * NOTE : use getNeighbor({0/1,0/1,0/1}) to iterate over subcells. 
   *        If block size is odd, you may need to use getNeighbor_ghost();
   **/
  KOKKOS_INLINE_FUNCTION
  CellIndex convert_index_ghost(const CellIndex& iCell) const;

  /**
   * Same as convert_index_ghost, but returns the subcell closest to the original cell when level_diff < 0
   * This has the same behaviour as calling getNeighbor_ghost() from a cell at the edge of the block 
   * with the offset corresponding to the ghost position
   * NOTE : read carefully the doc for getNeighbor_ghost(). The behavior when level_diff < 0 (smaller neighbor) might not be what is expected.
   *        This exists for compatibility with cell-based AMR, if your block size is wide enough, use convert_index_ghost() instead
   **/
  KOKKOS_INLINE_FUNCTION
  CellIndex convert_index_getNeighbor(const CellIndex& iCell) const;


  /**
   *  Get value of field for cell iCell
   * 
   * @param iCell must have a block size compatible with array
   **/
  KOKKOS_INLINE_FUNCTION
  real_t& at( const CellIndex& iCell, VarIndex field ) const;

  void exchange_ghosts(const GhostCommunicator& ghost_comm)
  {
    ghost_comm.exchange_ghosts(U, Ughost);
  }
};

template< typename View_t >
CellIndex CellArray_base<View_t>::convert_index(const CellIndex& in) const
{
  assert( in.is_valid() ); // Index needs to be valid for conversion
  assert( in.is_local() ); // cannot access ghosts in CellArray

  if( in.bx == bx && in.by == by && in.bz == bz )
    return in;

  int32_t gx = ((int32_t)bx-(int32_t)in.bx)/2;
  int32_t gy = ((int32_t)by-(int32_t)in.by)/2;
  int32_t gz = ((int32_t)bz-(int32_t)in.bz)/2;
  int32_t i = in.i + gx;
  int32_t j = in.j + gy;
  int32_t k = in.k + gz;

  assert(i>=0); assert(i<(int32_t)bx);
  assert(j>=0); assert(j<(int32_t)by);
  assert(k>=0); assert(k<(int32_t)bz);

  return CellIndex{in.iOct, (uint32_t)i, (uint32_t)j, (uint32_t)k, bx, by, bz, CellIndex::LOCAL_TO_BLOCK};
}

template< typename View_t >
CellIndex CellArray_base<View_t>::convert_index_ghost(const CellIndex& in) const
{
  assert( in.is_valid() ); // Index needs to be valid for conversion
  assert( in.is_local() ); // cannot access ghosts in CellArray

  if( in.bx == bx && in.by == by && in.bz == bz )
    return in;

  int32_t gx = ((int32_t)bx-(int32_t)in.bx)/2;
  int32_t gy = ((int32_t)by-(int32_t)in.by)/2;
  int32_t gz = ((int32_t)bz-(int32_t)in.bz)/2;
  uint32_t i = in.i + gx;
  uint32_t j = in.j + gy;
  uint32_t k = in.k + gz;

  if( i>=bx || j>=by || k>=bz )
    return CELLINDEX_INVALID;

  return CellIndex{in.iOct, i, j, k, bx, by, bz, CellIndex::LOCAL_TO_BLOCK};
}

KOKKOS_INLINE_FUNCTION
CellIndex convert_index_ghost_aux(const CellArray_global_ghosted& array, const CellIndex& in, CellIndex::offset_t& offset)
{
  assert( in.is_valid() ); // Index needs to be valid for conversion

  int32_t bx = array.bx;
  int32_t by = array.by;
  int32_t bz = array.bz;

  if( (int32_t)in.bx == bx && (int32_t)in.by == by && (int32_t)in.bz == bz )
    return in;
  
  int32_t gx = (bx-(int32_t)in.bx)/2;
  int32_t gy = (by-(int32_t)in.by)/2;
  int32_t gz = (bz-(int32_t)in.bz)/2;
  int32_t i = in.i + gx;
  int32_t j = in.j + gy;
  int32_t k = in.k + gz;

  if( i<0 || i>=bx || j<0 || j>=by || k<0 || k>=bz )
  {   // Index is outside of block : find neighbor
      // Closest position inside block
      uint32_t i_in = (uint32_t)i;
      uint32_t j_in = (uint32_t)j;
      uint32_t k_in = (uint32_t)k;
      // Offset from _in position
      int8_t di = 0;
      int8_t dj = 0;
      int8_t dk = 0;
      if(i<0)    { di=i   ; i_in=0;    }
      if(i>=bx)  { di=i-bx+1; i_in=bx-1; }
      if(j<0)    { dj=j   ; j_in=0;    }
      if(j>=by)  { dj=j-by+1; j_in=by-1; }
      if(k<0)    { dk=k   ; k_in=0;    }
      if(k>=bz)  { dk=k-bz+1; k_in=bz-1; }

      CellIndex iCell_in{in.iOct, i_in, j_in, k_in, (uint32_t)bx, (uint32_t)by, (uint32_t)bz};
      offset = {di,dj,dk};
      CellIndex iCell_out = iCell_in.getNeighbor_ghost( offset, array );
      return iCell_out;
  }
  else
  { // Index is inside block
    // non-local cells keep their non-local status, but not their level difference
    CellIndex::Status cell_status =  in.is_local()?
                                       CellIndex::LOCAL_TO_BLOCK
                                     : CellIndex::SAME_SIZE;
    return CellIndex{in.iOct, (uint32_t)i, (uint32_t)j, (uint32_t)k, (uint32_t)bx, (uint32_t)by, (uint32_t)bz, cell_status};
  }
}

CellIndex CellArray_global_ghosted::convert_index_getNeighbor(const CellIndex& in) const
{
  CellIndex::offset_t offset{};
  return convert_index_ghost_aux(*this, in, offset);
}

CellIndex CellArray_global_ghosted::convert_index_ghost(const CellIndex& in) const
{
  CellIndex::offset_t offset{};
  CellIndex iCell_n = convert_index_ghost_aux(*this, in, offset);
  int8_t offset_x = -(offset[IX]<0);
  int8_t offset_y = -(offset[IY]<0);
  int8_t offset_z = -(offset[IZ]<0);
  if( iCell_n.level_diff() == -1 && ( offset_x || offset_y || offset_z))
  {
    CellIndex iCell_res = iCell_n.getNeighbor({ offset_x, offset_y, offset_z });
    iCell_res.status = CellIndex::Status::SMALLER;
    return iCell_res;
  }
  else
  {
    return iCell_n;
  }
}

template< typename View_t >
real_t& CellArray_base<View_t>::at(const CellIndex& iCell, VarIndex field) const
{
  assert(bx == iCell.bx);
  assert(by == iCell.by);
  assert(bz == iCell.bz);

  uint32_t i = iCell.i + iCell.j*iCell.bx + iCell.k*iCell.bx*iCell.by;
  return U(i, fm[field], iCell.iOct.iOct%nbOcts);
}

real_t& CellArray_global_ghosted::at(const CellIndex& iCell, VarIndex field) const
{
  assert(bx == iCell.bx);
  assert(by == iCell.by);
  assert(bz == iCell.bz);

  uint32_t i = iCell.i + iCell.j*iCell.bx + iCell.k*iCell.bx*iCell.by;
  if( iCell.iOct.isGhost )
  {
    assert( Ughost.is_allocated() );
    return Ughost(i, fm[field], iCell.iOct.iOct);
  }
  else
  {
    return U(i, fm[field], iCell.iOct.iOct);
  }
}

CellIndex CellIndex::getNeighbor( const offset_t& offset ) const
{
  assert(is_valid());

  CellIndex res = *this;
  res.i += offset[IX];
  res.j += offset[IY];
  res.k += offset[IZ];
  res.status = res.is_local()                       
                ? CellIndex::LOCAL_TO_BLOCK
                : CellIndex::SAME_SIZE;

  assert(res.i < bx);
  assert(res.j < by);
  assert(res.k < bz);
 
  return res;
}

CellIndex CellIndex::getNeighbor_ghost( const offset_t& offset, const CellArray_global_ghosted& array ) const
{
  assert(this->is_valid());
  assert(this->bx == array.bx );
  assert(this->by == array.by );
  assert(this->bz == array.bz );
  assert(this->is_local() || this->level_diff() == -1);
  assert(int(this->bx) >= abs(offset[IX])*2 - 1 );
  assert(int(this->by) >= abs(offset[IY])*2 - 1 );
  assert(int(this->bz) >= abs(offset[IZ])*2 - 1 );

  const LightOctree& lmesh = array.lmesh;

  int32_t i = this->i + offset[IX];
  int32_t j = this->j + offset[IY];
  int32_t k = this->k + offset[IZ];

  LightOctree::offset_t oct_offset{
    (int8_t)std::floor( (float)i/(float)bx ),
    (int8_t)std::floor( (float)j/(float)by ),
    (int8_t)std::floor( (float)k/(float)bz )
  };

  if( oct_offset[IX] == 0 && oct_offset[IY] == 0 && oct_offset[IZ] == 0 )
  { // Neighbor cell is inside local octant
    CellIndex res = this->getNeighbor( offset );
    assert(res.is_valid());
    return res;
  }
  else
  { // Neighbor cell is outside local octant : need to find cell in neighbor octant
    const LightOctree::OctantIndex& iOct = this->iOct; 
    //assert(!iOct.isGhost);
    if( lmesh.isBoundary( iOct, oct_offset ) )
    {
      return CELLINDEX_BOUNDARY;
    }
    
    LightOctree::NeighborList oct_neighbors = lmesh.findNeighbors(iOct, oct_offset);
    assert( oct_neighbors.size() != 0 ); 

    int level_diff = lmesh.getLevel(iOct) - lmesh.getLevel(oct_neighbors[0]);
    
    // Compute position of cell in neighbor when neighbor is same size;
    uint32_t i_same = i - oct_offset[IX] * bx;
    uint32_t j_same = j - oct_offset[IY] * by;
    uint32_t k_same = k - oct_offset[IZ] * bz;
    
    if( level_diff == 0 )
    { // Neighbor is same size
      assert(i_same<bx);
      assert(j_same<by);
      assert(k_same<bz);
      return CellIndex{
        oct_neighbors[0],
        i_same, j_same, k_same,
        bx, by, bz,
        CellIndex::SAME_SIZE
      };
    }
    else if(level_diff == 1)
    { // Neighbor is larger  

      // Compute suboctant where target cell is located in larger neighbor
      LightOctree::pos_t current_center = lmesh.getCenter(iOct);
      real_t current_size = lmesh.getSize(iOct);

      int current_logical_x = std::floor( current_center[IX]/current_size );
      int current_logical_y = std::floor( current_center[IY]/current_size );
      int current_logical_z = std::floor( current_center[IZ]/current_size );

      auto is_odd = [](int x) {
        return (int)(x%2 != 0);
      };

      int suboctant_offset_x = is_odd( current_logical_x + oct_offset[IX] );
      int suboctant_offset_y = is_odd( current_logical_y + oct_offset[IY] );
      int suboctant_offset_z = is_odd( current_logical_z + oct_offset[IZ] );

      // Offset to select suboctant and /2 for position in larger octant 
      uint32_t i_larger = (i_same+suboctant_offset_x*bx)/2;
      uint32_t j_larger = (j_same+suboctant_offset_y*by)/2;
      uint32_t k_larger = (k_same+suboctant_offset_z*bz)/2;

      assert(i_larger<bx);
      assert(j_larger<by);
      assert(k_larger<bz);

      CellIndex res{
        oct_neighbors[0], 
        i_larger, j_larger, k_larger,
        bx, by, bz,
        CellIndex::BIGGER
      }; 

      return res;    
    }
    else if(level_diff == -1)
    { // Neighbor is smaller : compute CellIndex of "first neighbor" (neighbor cell closest to origin)

      // Compute cell position in neighbor meta-bloc of size {2*bx, 2*by, 2*bz}
      // Pick the smallest index {i_smaller, j_smaller, k_smaller} contiguous to current cell
      uint32_t i_smaller = i*2 + (int)(i<0) - oct_offset[IX] * bx * 2; 
      uint32_t j_smaller = j*2 + (int)(j<0) - oct_offset[IY] * by * 2;
      uint32_t k_smaller = k*2 + (int)(k<0) - oct_offset[IZ] * bz * 2;

      assert(i_smaller < 2*bx );
      assert(j_smaller < 2*by );
      assert(k_smaller < 2*bz );

      // Compute position of suboctant containing "first neighbor" among the 8 suboctants
      int suboctant_x = i_smaller >= bx;    
      int suboctant_y = j_smaller >= by;    
      int suboctant_z = k_smaller >= bz;

      // Shift cell index to appropriate suboctant
      i_smaller -= bx * suboctant_x;
      j_smaller -= by * suboctant_y;
      k_smaller -= bz * suboctant_z;

      assert(i_smaller < bx );
      assert(j_smaller < by );
      assert(k_smaller < bz );

      // Find suboctant containing first neighbor
      LightOctree::pos_t current_oct_center = lmesh.getCenter(iOct);
      real_t current_oct_size = lmesh.getSize(iOct);
      LightOctree::pos_t neighbor_superoct_center{
        current_oct_center[IX] + oct_offset[IX] * current_oct_size, 
        current_oct_center[IY] + oct_offset[IY] * current_oct_size, 
        current_oct_center[IZ] + oct_offset[IZ] * current_oct_size 
      };
      int suboctant = -1;
      for( size_t i=0; i<oct_neighbors.size(); i++ )
      {
        LightOctree::pos_t neighbor_suboct_center = lmesh.getCenter(oct_neighbors[i]);
        // Compute position of suboctant in bigger neighbor octant
        int this_suboctant_x = neighbor_suboct_center[IX] > neighbor_superoct_center[IX];
        int this_suboctant_y = neighbor_suboct_center[IY] > neighbor_superoct_center[IY];
        int this_suboctant_z = neighbor_suboct_center[IZ] > neighbor_superoct_center[IZ];

        // Match suboctant with suboctant containing first neighbor cell
        if( suboctant_x == this_suboctant_x && suboctant_y == this_suboctant_y && suboctant_z == this_suboctant_z )
        {
          suboctant = i;
          break;
        }
      }

      assert( suboctant != -1 );

      return CellIndex{
        oct_neighbors[suboctant], 
        (uint32_t)i_smaller, (uint32_t)j_smaller, (uint32_t)k_smaller,
        bx, by, bz,
        CellIndex::SMALLER};
    }
    else
    {
      assert(false); // Level-diff doesn't respect 2:1 balance
    }
  }

  assert(false); // unhandled case
  return CELLINDEX_INVALID;
}

template< typename View_t >
CellIndex CellIndex::getNeighbor_ghost( const offset_t& offset, const CellArray_base<View_t>& array ) const
{
  assert(this->is_valid());
  assert(this->bx == array.bx );
  assert(this->by == array.by );
  assert(this->bz == array.bz );
  assert(this->is_local());

  uint32_t i = this->i + offset[IX];
  uint32_t j = this->j + offset[IY];
  uint32_t k = this->k + offset[IZ];

  if( i>=bx || j>=by || k>=bz )
  { // Neighbor cell is inside local octant
    CellIndex res = this->getNeighbor( offset );
    assert(res.is_valid() && res.is_local());
    return res;
  }
  else
  { 
    return CELLINDEX_INVALID;
  }

}

} // namespace CellArray_impl

} // namespace dyablo