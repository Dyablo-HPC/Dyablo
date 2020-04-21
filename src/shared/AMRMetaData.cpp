#include "shared/AMRMetaData.h"

#include "shared/morton_utils.h"

namespace dyablo
{

// =============================================
// ==== CLASS AMRMetaData IMPL =================
// =============================================


// =============================================
// =============================================
AMRMetaData::AMRMetaData(uint64_t capacity) :
  m_hashmap(capacity),
  m_capacity(capacity)
{

} // AMRMetaData::AMRMetaData

// =============================================
// =============================================
AMRMetaData::~AMRMetaData()
{
} // AMRMetaData::~AMRMetaData

// =============================================
// =============================================
void AMRMetaData::update(const AMRmesh& mesh)
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
        uint8_t dim = mesh.getDim();

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

} // AMRMetaData::update

// =============================================
// =============================================
void AMRMetaData::report()
{

  std::cout << "AMRMetaData hashmap size     = " << m_hashmap.size() << std::endl;
  std::cout << "AMRMetaData hashmap capacity = " << m_hashmap.capacity() << " (max size)" << std::endl;

} // AMRMetaData::update

} // namespace dyablo
