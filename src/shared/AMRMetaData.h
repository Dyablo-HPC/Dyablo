#ifndef DYABLO_AMR_METADATA_H_
#define DYABLO_AMR_METADATA_H_

#include "shared/kokkos_shared.h"
#include "shared/amr_key.h" // for amr_key_t typedef
#include "shared/bitpit_common.h"  // for type AMRmesh = bitpit::PabloUniform

#include "shared/morton_utils.h" // for compute_morton_key

#include <Kokkos_UnorderedMap.hpp>


namespace dyablo 
{

/**
 * Class storing AMR geometric data and cell connectivity,
 * decouple from bitpit and usable in portable kokkos functors.
 *
 * See ideas of design
 * https://gitlab.maisondelasimulation.fr/pkestene/dyablo/issues/42
 *
 * This class holds a meta data container, a hashmap on device (in Kokkos
 * sens).
 * 
 */
template<int dim>
class AMRMetaData
{

public:
  static constexpr int m_dim = dim;

  //! hashmap type alias for key
  using key_t = amr_key_t;

  //! hashmap type alias for value
  using value_t = uint64_t;

  //! hashmap type alias
  using hashmap_t = Kokkos::UnorderedMap<key_t, value_t, dyablo::Device>;

  //! type used to stored neighbor level status
  //! we need more bits to encode status in 3D, see below
  using neigh_level_status_t = typename std::conditional<dim==2,uint16_t,uint64_t>::type;

  //! type alias for the array storing neighbor level status
  using neigh_level_status_array_t = Kokkos::View<neigh_level_status_t*, dyablo::Device>;

  //! type alias 
  using neigh_rel_pos_status_t = typename std::conditional<dim==2,uint8_t,uint32_t>::type; 

  //! type alias for the array storing neighbor relative position
  using neigh_rel_pos_status_array_t = Kokkos::View<neigh_rel_pos_status_t*, dyablo::Device>;

  /**
   * enum for representing relative level difference between
   * current octant and neighbor octant (accross a face, and edge or
   * corner).
   *
   * this enum is uint8_t but we actually only need 2 bits to store
   * information.
   *
   * Reminder : small level means large octant
   *
   * NEIGH_IS_LARGER  means neighbor octant has a smaller level
   * NEIGH_IS_SMALLER means neighbor octant has a larger level
   */
  enum NEIGHBOR_LEVEL : neigh_level_status_t 
  {
    NEIGH_IS_SAME_SIZE       = 0x0, /* 0 */
    NEIGH_IS_SMALLER         = 0x1, /* at level+1 */
    NEIGH_IS_EXTERNAL_BORDER = 0x2,
    NEIGH_NONE               = 0x2, /* no neighbor (e.g. across a corner touching a hanging face)*/
    NEIGH_IS_LARGER          = 0x3  /* at level-1 using 2's complement notation */
  };

  std::string neighbor_level_to_string(NEIGHBOR_LEVEL nl)
  {
    if(nl == 0x0) 
      return "NEIGH_IS_SAME";
    else if (nl == 0x1)
      return "NEIGH_IS_SMALLER";
    else if (nl == 0x2)
      return "NEIGH_NONE";
    else
      return "NEIGH_IS_LARGER";
  }

  /**
   * - in 2D, across a face there can only be 2 neighbors, so only
   * RELATIVE_POS_0 and RELATIVE_POS_1 will be used.
   * - in 3D, there can be up to 4 neighbors across a face,
   *  and 2 neighbors across an edge.
   *
   * Total number of bits required:
   * in 2D : 4 faces x 1 bit = 4 bits
   * in 3D : 6 faces x 2 bits + 12 edges x 1 bit = 24 bits
   */
  enum RELATIVE_POSITION : neigh_rel_pos_status_t
  {
    RELATIVE_POS_0 = 0x0,
    RELATIVE_POS_1 = 0x1,
    RELATIVE_POS_2 = 0x2,
    RELATIVE_POS_3 = 0x3
  };


  //! number of bits required to stored neighbor level status
  static constexpr int NEIGH_BIT_WIDTH = 2; 

  //! constructor 
  AMRMetaData(uint64_t capacity);
  
  //! destructor
  ~AMRMetaData();

  /**
   * update and/or resize hashmap with data from a given pablo object.
   *
   * \param[in] pablo object
   */
  void update_hashmap(const AMRmesh& mesh);

  /**
   * update neighbor level status.
   *
   * For each regular octant, encode neighbor level status.
   * This method will fill Kokkos view m_neigh_level_status
   * of size m_nbOctants.
   *
   * The data type of m_neigh_level_status view is uint64_t, indeed:
   * - to encode level (-1, 0, +1) : 2 bits for each type of neighbor
   * - in 2D :
   * - there are 4 faces in 2D   --> 4x2 =  8 bits
   * - there are 4 corners in 2D --> 4x2 =  8 bits
   *   total is                            16 bits
   *
   * - in 3D :
   * - there are 6 faces in 3D   --> 6x2  = 12 bits
   * - there are 8 corners in 3D --> 8x2  = 16 bits
   * - there are 12 edges in 3D  --> 12x2 = 24 bits
   *   total is                             52 bits (rounded up to 64bits)
   *   Note that is edge information is not really necessary, a 32 bits
   *   variable is sufficient.
   *
   * To store all this information, we need a total number of bits
   * different for 2D or 3D, so we defined an templated alias for
   * that, see neigh_status_t
   */
  void update_neigh_level_status(const AMRmesh& mesh);

  /**
   * Update array m_neigh_rel_pos_status.
   *
   * This array contains one neigh_status_t value per octant
   * all octant are concerned (regular and ghost).
   *
   * for all octant
   *   for all interface (face, edge) // corners are not concerned
   *      if ( (neigh is same size) or (neigbor is smaller)
   *         return 0;
   *      else // neighbor is larger
   *         encode relative position of current octant among
   *         all octants touching the same neighbor (see below)
   *
   * =============================================================
   * E.g. small cell is current octant, large one is the neighbor 
   * when dir = DIR_X and face = FACE_RIGHT (neighbor is on the right):
   *
   * RELATIVE_POS_0          RELATIVE_POS_1
   *        ______           __      ______ 
   *       |      |         |1 |    |      |
   *  __   |      |     or  |__|    |      |
   * |0 |  |      |                 |      |
   * |__|  |______|                 |______|
   *
   * when dir = DIR_Y and face = FACE_RIGHTT (neigbor is on the right
   * along Y direction):
   *
   * RELATIVE_POS_0       RELATIVE_POS_1
   *  ______               ______ 
   * |      |             |      |
   * |      |         or  |      |
   * |      |             |      |
   * |______|             |______|
   *
   *  __                       __
   * |0 |                     |1 |
   * |__|                     |__|
   *

   * \note in 3D, there are 4 relative positions, ordered using Morton 
   * order
   */
  void update_neigh_rel_pos_status(const AMRmesh& mesh);

  //! minimal reporting
  void report();

  //! get hashmap capacity
  uint64_t capacity() { return m_capacity; }

  //! get hashmap
  const hashmap_t& hashmap() { return m_hashmap; }

  //! get neighborlevel_status array
  const neigh_level_status_array_t& neigh_level_status_array() 
  { 
    return m_neigh_level_status;
  }

  //! helper for debug / testing : decode neigh status
  void decode_neighbor_status(uint64_t iOct);

  //! get neigh_rel_pos_status array
  const neigh_rel_pos_status_array_t& neigh_rel_pos_status_array() 
  { 
    return m_neigh_rel_pos_status;
  }

private:
  //! main metadata container, an unordered map 
  //! key is Morton index, value is memory/iOct index
  hashmap_t m_hashmap;

  //! neighbor level status array
  neigh_level_status_array_t m_neigh_level_status;

  //! neighbor relative position array
  neigh_rel_pos_status_array_t m_neigh_rel_pos_status;

  //! hashmap capacity
  uint64_t m_capacity;

  //! AMR number of regular octants (current MPI process)
  //! changed each time update_hashmap() method is called
  uint64_t m_nbOctants;

  //! AMR number of ghosts octants (current MPI process)
  //! changed each time update_hashmap() method is called
  uint64_t m_nbGhosts;

}; // class AMRMetaData

// =============================================
// ==== CLASS AMRMetaData IMPL =================
// =============================================

// =============================================
// =============================================
template<int dim>
AMRMetaData<dim>::AMRMetaData(uint64_t capacity) :
  m_hashmap(capacity),
  m_neigh_level_status(),
  m_neigh_rel_pos_status(),
  m_capacity(capacity)
{

} // AMRMetaData::AMRMetaData

// =============================================
// =============================================
template<int dim>
AMRMetaData<dim>::~AMRMetaData()
{
} // AMRMetaData::~AMRMetaData

// =============================================
// =============================================
template<int dim>
void AMRMetaData<dim>::update_hashmap(const AMRmesh& mesh)
{

  m_nbOctants = mesh.getNumOctants();
  m_nbGhosts  = mesh.getNumGhosts();

  // check if hashmap needs a rehash
  if (m_nbOctants + m_nbGhosts > 0.75*m_capacity)
  {
    // increase capacity
    m_capacity = m_capacity * 2;

    m_hashmap.rehash(m_capacity);

    std::cout << "hashmap needs to be rehashed.\n";
    std::cout << "hashmap new capacity is " << m_hashmap.capacity() << ".\n";
  }

  // create a mirror on host
  hashmap_t::HostMirror hashmap_host(m_hashmap.capacity());

  // insert data from pablo mesh object, do that on host, with OpenMP
  {
    Kokkos::RangePolicy<Kokkos::OpenMP> policy(0, m_nbOctants+m_nbGhosts);
    Kokkos::parallel_for(
      policy,
      [&](const uint64_t iOct)
      {
        amr_key_t key;

        // the following way of computing Morton index
        // is exactly identical to bitpit
        uint32_t x,y,z;

        if (iOct < m_nbOctants)
        {
          // we have a regular octant

          x = mesh.getOctant(iOct)->getLogicalX();
          y = mesh.getOctant(iOct)->getLogicalY();
          z = mesh.getOctant(iOct)->getLogicalZ();

        }
        else
        {
          // we have a ghost octant

          int64_t iOctG = iOct-m_nbOctants;
          x = mesh.getGhostOctant(iOctG)->getLogicalX();
          y = mesh.getGhostOctant(iOctG)->getLogicalY();
          z = mesh.getGhostOctant(iOctG)->getLogicalZ();
          
        }

        uint64_t morton_key = compute_morton_key(x,y,z);

        // Morton index a la bitpit
        //key[0] = mesh.getMorton(iOct); /* octant's morton index */

        // Morton index a la dyablo
        key[0] = morton_key;

        if (iOct < m_nbOctants)
        {
          key[1] = mesh.getLevel(iOct);  /* octant's level */
        }
        else
        {
          int64_t iOctG = iOct-m_nbOctants;

          key[1] = mesh.getGhostOctant(iOctG)->getLevel();
        }

        value_t value = iOct;

        hashmap_host.insert( key, value );

      });
  }

  // finaly update the new hashmap on device
  Kokkos::deep_copy(m_hashmap, hashmap_host);

} // AMRMetaData::update_hashmap

// =============================================
// =============================================
template<int dim>
void AMRMetaData<dim>::update_neigh_level_status(const AMRmesh& mesh)
{
} // update_neigh_level_status

// declare specializations
template<>
void AMRMetaData<2>::update_neigh_level_status(const AMRmesh& mesh);
template<>
void AMRMetaData<3>::update_neigh_level_status(const AMRmesh& mesh);

// =============================================
// =============================================
template<int dim>
void AMRMetaData<dim>::decode_neighbor_status(uint64_t iOct) 
{

  const uint8_t nbFaces   = 2*m_dim;
  const uint8_t nbCorners = dim==2 ? 4 : 8;

  neigh_level_status_t status = m_neigh_level_status(iOct);
  
  std::cout << "Neigh status of iOct = " << iOct << " :\n";
  for (int iface=0; iface<nbFaces; ++iface)
  {
    NEIGHBOR_LEVEL nl = static_cast<NEIGHBOR_LEVEL>( (status >> (2*iface)) & 0x3 );

    std::cout << "face = " << iface << " status is "
              << neighbor_level_to_string(nl) << "\n";

  }

  for (int icorner=0; icorner<nbCorners; ++icorner)
  {
    int icorner2 = icorner + nbFaces;

    NEIGHBOR_LEVEL nl = static_cast<NEIGHBOR_LEVEL>( (status >> (2*icorner2)) & 0x3 );

    std::cout << "corner = " << icorner << " status is "
              << neighbor_level_to_string(nl) << "\n";

  }

  if (m_dim==3)
  {
    // TODO : print something about edges
  }

}; // AMRMetaData<dim>::decode_neighbor_status

// =============================================
// =============================================
template<int dim>
void AMRMetaData<dim>::update_neigh_rel_pos_status(const AMRmesh& mesh)
{
} // update_neigh_rel_pos_status

// declare specializations
template<>
void AMRMetaData<2>::update_neigh_rel_pos_status(const AMRmesh& mesh);
template<>
void AMRMetaData<3>::update_neigh_rel_pos_status(const AMRmesh& mesh);


// =============================================
// =============================================
template<int dim>
void AMRMetaData<dim>::report()
{

  std::cout << "AMRMetaData hashmap size     = " << m_hashmap.size() << std::endl;
  std::cout << "AMRMetaData hashmap capacity = " << m_hashmap.capacity() << " (max size)" << std::endl;

} // AMRMetaData::report

} // namespace dyablo

#endif // DYABLO_AMR_METADATA_H_
