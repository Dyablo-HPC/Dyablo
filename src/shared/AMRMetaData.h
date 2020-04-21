#ifndef DYABLO_AMR_METADATA_H_
#define DYABLO_AMR_METADATA_H_

#include "shared/kokkos_shared.h"
#include "shared/amr_key.h" // for amr_key_t typedef
#include "shared/bitpit_common.h"  // for type AMRmesh = bitpit::PabloUniform

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
class AMRMetaData
{

public:
  using key_t = amr_key_t;
  using value_t = uint64_t;
  using hashmap_t = Kokkos::UnorderedMap<key_t, value_t, dyablo::Device>;

  //! constructor 
  AMRMetaData(uint64_t capacity);
  
  //! destructor
  ~AMRMetaData();

  /**
   * update and/or resize hashmap with data from a given pablo object.
   *
   * \param[in] pablo object
   */
  void update(const AMRmesh& mesh);

  //! minimal reporting
  void report();

  //! get hashmap capacity
  uint64_t capacity() { return m_capacity; }

  //! get hashmap
  const hashmap_t& hashmap() { return m_hashmap; }

private:
  //! main metadata container
  hashmap_t m_hashmap;

  //! hashmap capacity
  uint64_t m_capacity;

  //! AMR number of regular octants (current MPI process)
  //! changed each time update() method is called
  uint64_t m_nbOctants;

  //! AMR number of ghosts octants (current MPI process)
  //! changed each time update() method is called
  uint64_t m_nbGhosts;

}; // class AMRMetaData

} // namespace dyablo

#endif // DYABLO_AMR_METADATA_H_
