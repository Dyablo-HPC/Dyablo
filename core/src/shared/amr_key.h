/**
* \file amr_key.h
* \brief
 * Design a "key" type to be used as entry to our adaptive mesh refinement
 * hash-table data structure.
 *
 * Let's try two types of key:
 * - morton index + tree
 * - morton index + level + tree
 *
 * The first type should be usefull to store all the leaves of tree.
 * The second type could be usefull to store additionally level ghost cells.
 *   Indeed this allows to have cells of different sizes covering the same
 *   location (same x,y,z but different levels).
 *
 * \note When the AMR tree is a PABLO object from bitpit, there is only one
 * tree, but in case one day we would like to test something as p4est,
 * let's also deal with the multi-tree case.
 */
#ifndef DYABLO_SHARED_KEY_TYPE_H
#define DYABLO_SHARED_KEY_TYPE_H

#include <cstdint>

namespace dyablo
{

/**
 * Cell status.
 *
 * - "ghost level cells" are inserted in hashmap, to make any regular
 *   cell have a regular neighborhood; they are inserted by interpolated
 *   the payload data.
 *
 * - "ghost external cell" hold the border condition; it inserted in the
 * hashmap when visiting a cell that "touches the external border" and that
 * hold the very same Morton key, same level, only the payload data is
 * different
 *
 * \note Not used when AMR is done with bitpit
 */
enum CellStatus : uint8_t
{

  CELL_INVALID = 0,
  CELL_REGULAR = 1,       /*!< a regular cell */
  CELL_GHOST_LEVEL = 2,   /*!< a cell inserted in hashmap to make neighbor cell surrounded by cells that have the same level */
  CELL_GHOST_MPI = 3,     /*!< an MPI ghost cell */
  CELL_GHOST_EXTERNAL = 4,/*!< a cell outside the external domain */
  CELL_TO_BE_REMOVED = 5, /*!< this cell should be removed soon */
  CELL_UNINITIALIZED = 6  /*!< this cell should be initialed soon */

}; // enum CellStatus

/**
 * \struct amr_key_t
 *
 * Define the key type as entry to our Kokkos::UnorderedMap data structure
 * using morton index + level + tree id.
 *
 * keys[0] holds the morton index computed inside the tree the cell 
 *         belongs to.
 * keys[1] holds additionnal information:
 *  - the last 8 bits encode the level (only 5 bits is required, 
 *    since in 2D MAXLEVEL is 29).
 *  - the next 16 LSB encode the tree id; so only 2^16=65536 trees are 
 *    allowed for now, but it should sufficient for most application
 *  - the remaining bits are unused
 *
 * Just for clarification, this key type is not a morton key since here
 * cell level is encoded.
 *
 * \sa amr_key_simple_t
 */
struct amr_key_t
{

  /** bit mask to extract LEVEL (the last 8 bits) */
  static constexpr uint64_t LEVEL_MASK = 0x00000000000000FF;

  /** bit mask to extract TREEID (16 bits) */
  static constexpr uint64_t TREE_ID_MASK = 0x0000000000FFFF00;

  uint64_t keys[2];

  KOKKOS_INLINE_FUNCTION
  amr_key_t() : keys{0,0} {}

  KOKKOS_INLINE_FUNCTION
  amr_key_t(uint64_t key1, uint64_t key2) : keys{key1,key2} {}

  KOKKOS_INLINE_FUNCTION
  uint64_t operator[](size_t i) const { return keys[i]; }

  KOKKOS_INLINE_FUNCTION
  uint64_t& operator[](size_t i) { return keys[i]; }

  /** extract Morton key */
  KOKKOS_INLINE_FUNCTION
  uint64_t get_morton()
  {
    return keys[0];
  } // get_morton

  /** extract level */
  KOKKOS_INLINE_FUNCTION
  uint8_t get_level()
  {
    return (keys[1] & LEVEL_MASK); 
  } // get_level
  
  /** extract treeId */
  KOKKOS_INLINE_FUNCTION
  uint16_t get_treeId()
  {
    return (keys[1] & TREE_ID_MASK) >> 8; 
  } // get_treeid

  KOKKOS_INLINE_FUNCTION
  bool operator < (const amr_key_t& other) const
  {

    return (keys[1] < other.keys[1]) or (keys[1]==other.keys[1] and keys[0] < other.keys[0]);

  }
  
  KOKKOS_INLINE_FUNCTION
  bool operator == (const amr_key_t& other) const
  {

    return (keys[1] == other.keys[1]) and (keys[0]==other.keys[0]);

  }
  
}; // struct amr_key_t

/** encode level and treeId in an uint64_t integer */
KOKKOS_INLINE_FUNCTION
static uint64_t encode_level_tree(int level, int treeId = 0)
{
  
  uint64_t res = 0;
  
  // first 8 bits for level
  res = (level & amr_key_t::LEVEL_MASK);
  
  // then 16 bits for treeId
  res = res | ( (treeId << 8) & amr_key_t::TREE_ID_MASK);
  
  return res;
  
} // encode_level_tree


/**
 * \struct morton_key_t
 *
 * Define the key type as entry to our Kokkos::UnorderedMap data structure
 * using morton index + tree id.
 *
 * keys[0] holds the morton index computed inside the tree the cell 
 *         belongs to.
 * keys[1] holds additionnal information:
 *  - the last 16 least significant bits  encode the tree id; 
 *    so only 2^16=65536 trees are allowed
 *    for now, but it should sufficient for most application
 *  - the remaining bits are unused
 */
struct morton_key_t
{
  /** bit mask to extract TREEID (the last 16 bits) */
  static constexpr uint64_t TREE_ID_MASK = 0x000000000000FFFF;

  uint64_t keys[2];
  
  KOKKOS_INLINE_FUNCTION
  morton_key_t() : keys{0,0} {}
  
  KOKKOS_INLINE_FUNCTION
  morton_key_t(uint64_t key1, uint64_t key2) : keys{key1,key2} {}
  
  KOKKOS_INLINE_FUNCTION
  uint64_t operator[](size_t i) const { return keys[i]; }
  
  KOKKOS_INLINE_FUNCTION
  uint64_t& operator[](size_t i) { return keys[i]; }

  /** extract Morton key */
  KOKKOS_INLINE_FUNCTION
  uint64_t get_morton()
  {
    return keys[0];
  } // get_morton
  
  /** extract treeId */
  KOKKOS_INLINE_FUNCTION
  uint16_t get_treeId()
  {
    return (keys[1] & TREE_ID_MASK); 
  } // get_treeid
  
}; // struct morton_key_t

/** encode treeId in an uint64_t integer */
KOKKOS_INLINE_FUNCTION
static uint64_t encode_tree(int treeId)
{
  
  uint64_t res = 0;
  
  // 16 bits for level
  res = (treeId & morton_key_t::TREE_ID_MASK);
  
  return res;
  
} // encode_tree

} // namespace dyablo

#endif // DYABLO_SHARED_KEY_TYPE_H
