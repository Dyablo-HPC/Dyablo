#include "shared/AMRMetaData.h"

#include <iostream>

namespace dyablo
{

// =============================================
// ==== CLASS AMRMetaData IMPL =================
// =============================================

// =============================================
// =============================================
// template specialization for 2D
template<>
void AMRMetaData<2>::update_neighbor_status(const AMRmesh& mesh)
{

  const uint8_t nbFaces = 2*m_dim;
  
  const uint8_t nbCorners = m_dim == 2 ? 4 : 8;

  // resize to current number of regular octants
  Kokkos::resize(m_neigh_level_status, m_nbOctants);

  // create a mirror on host
  neighbor_level_status_t::HostMirror neigh_level_status_host = 
    Kokkos::create_mirror_view(m_neigh_level_status);

  // fill the mirrored array on host with OpenMP exec space
  {
    Kokkos::RangePolicy<Kokkos::OpenMP> policy(0, m_nbOctants);
    Kokkos::parallel_for(
      policy,
      [&](const uint64_t iOct)
      {
        neigh_status_t status = 0;

        /*
         * 1. face neighbors
         */
        const uint8_t codim = 1;

        for (int iface=0; iface<nbFaces; ++iface)
        {
          
          // list of neighbors octant id, neighbor through a given face
          std::vector<uint32_t> neigh;
          std::vector<bool> isghost;
          
          // ask PABLO to find neighbor octant id accross a given face
          // this fill vector neigh and isghost
          mesh.findNeighbours(iOct, iface, codim, neigh, isghost);
          
          // no neighbors means current octant is touching external border 
          if (neigh.size() == 0)
          {
            status |= (NEIGH_IS_EXTERNAL_BORDER << (2*iface) );
          }

          // 1 neighbor means neighbor is larger or same size
          else if (neigh.size() == 1) // neighbor is larger or same size
          {
            
            // retrieve neighbor octant id
            uint32_t iOct_neigh = neigh[0];

            uint8_t level = mesh.getLevel(iOct);
            uint8_t level_neigh = isghost[0] ?
              mesh.getGhostOctant(iOct_neigh)->getLevel() : // neigh is a MPI ghost
              mesh.getLevel(iOct_neigh);                    // neigh is a regular oct

            // neighbor is larger
            if ( level_neigh < level )
            {
              status |= (NEIGH_IS_LARGER << 2*iface);
            }

            // neighbor has same size
            else
            {
              status |= (NEIGH_IS_SAME_SIZE << 2*iface);
            }
            
          }

          // more neighbors means neighbors are smaller
          else if (neigh.size() > 1)
          {
            status |= (NEIGH_IS_SMALLER << 2*iface);
          }
          
        } // end for iface
        
        /*
         * 2. corner neighbors
         */
        const uint8_t corner_codim = 2;

        for (int icorner=0; icorner<nbCorners; ++icorner)
        {

          // icorner is used when bit shifting is involved
          int icorner2 = icorner + nbFaces;

          // list of neighbors octant id, neighbor through a given face
          std::vector<uint32_t> neigh;
          std::vector<bool> isghost;
          
          // ask PABLO to find neighbor octant id across a given corner
          // this fill vector neigh and isghost
          mesh.findNeighbours(iOct, icorner, corner_codim, neigh, isghost);

          // no neighbors means 2 things : current octant is
          // either touching external border
          // either the corner is at hanging face (meaning neighbor is larger)
          // still in both case, we use the same "code" NEIGH_NONE
          //
          // if you want to really know if a corner is "touching" external border
          // you just need to test status with the corresponding 2 adjacent faces
          // corner2  face 3  corner 3
          //        +-------+
          //        |       |
          // face 0 |       | face 1
          //        |       |
          //        +-------+
          // corner0  face 2  corner 1
          //
          if (neigh.size() == 0)
          {
            status |= (NEIGH_NONE << 2*icorner2);
          }

          // 1 neighbor 
          else if (neigh.size() == 1)
          {
            
            // retrieve neighbor octant id
            uint32_t iOct_neigh = neigh[0];

            uint8_t level       = mesh.getLevel(iOct);
            uint8_t level_neigh = isghost[0] ?
              mesh.getGhostOctant(iOct_neigh)->getLevel() : // neigh is a MPI ghost
              mesh.getLevel(iOct_neigh);                    // neigh is a regular oct

            // neighbor is larger
            if ( level_neigh < level )
            {
              status |= (NEIGH_IS_LARGER << 2*icorner2);
            }

            // neighbor has same size
            else if ( level_neigh == level )
            {
              status |= (NEIGH_IS_SAME_SIZE << 2*icorner2);
            }

            // neighbor is smaller
            else
            {
              status |= (NEIGH_IS_SMALLER << 2*icorner2);
            }
            
          }
                    
        } // end for icorner

        neigh_level_status_host(iOct) = status;

      }); // end Kokkos parallel_for

  } // end fill the mirrored array

  // copy array on device
  Kokkos::deep_copy (m_neigh_level_status, neigh_level_status_host);

} // AMRMetaData<2>::update_neighbor_status

// =============================================
// =============================================
// template specialization for 3D
template<>
void AMRMetaData<3>::update_neighbor_status(const AMRmesh& mesh)
{

  const uint8_t nbFaces = 2*m_dim;
  
  const uint8_t nbCorners = m_dim == 2 ? 4 : 8;

  // resize to current number of regular octants
  Kokkos::resize(m_neigh_level_status, m_nbOctants);
  
  // create a mirror on host
  neighbor_level_status_t::HostMirror neigh_level_status_host = 
    Kokkos::create_mirror_view(m_neigh_level_status);

  // fill the mirrored array on host with OpenMP exec space
  {
    Kokkos::RangePolicy<Kokkos::OpenMP> policy(0, m_nbOctants);
    Kokkos::parallel_for(
      policy,
      [&](const uint64_t iOct)
      {

        neigh_status_t status = 0;

        /*
         * 1. face neighbors
         */
        const uint8_t codim = 1;

        for (int iface=0; iface<nbFaces; ++iface)
        {
          
          // list of neighbors octant id, neighbor through a given face
          std::vector<uint32_t> neigh;
          std::vector<bool> isghost;
          
          // ask PABLO to find neighbor octant id accross a given face
          // this fill vector neigh and isghost
          mesh.findNeighbours(iOct, iface, codim, neigh, isghost);
          
          // no neighbors means current octant is touching external border 
          if (neigh.size() == 0)
          {
            status |= (NEIGH_IS_EXTERNAL_BORDER << 2*iface);
          }
          
          // 1 neighbor means neighbor is larger or same size
          else if (neigh.size() == 1) // neighbor is larger or same size
          {
            
            // retrieve neighbor octant id
            uint32_t iOct_neigh = neigh[0];
            
            uint8_t level = mesh.getLevel(iOct);
            uint8_t level_neigh = isghost[0] ?
              mesh.getGhostOctant(iOct_neigh)->getLevel() : // neigh is a MPI ghost
              mesh.getLevel(iOct_neigh);                    // neigh is a regular oct
            
            // neighbor is larger
            if ( level_neigh < level )
            {
              status |= (NEIGH_IS_LARGER << 2*iface);
            }
            
            // neighbor has same size
            else
            {
              status |= (NEIGH_IS_SAME_SIZE << 2*iface);
            }
            
          }
          
          // more neighbors means neighbors are smaller
          else if (neigh.size() > 1)
          {
            status |= (NEIGH_IS_SMALLER << 2*iface);
          }
          
        } // end for iface

        /*
         * 2. corner neighbors
         */
        const uint8_t corner_codim = 2;

        for (int icorner=0; icorner<nbCorners; ++icorner)
        {

          // icorner is used when bit shifting is involved
          int icorner2 = icorner + nbFaces;

          // list of neighbors octant id, neighbor through a given face
          std::vector<uint32_t> neigh;
          std::vector<bool> isghost;
          
          // ask PABLO to find neighbor octant id across a given corner
          // this fill vector neigh and isghost
          mesh.findNeighbours(iOct, icorner, corner_codim, neigh, isghost);

          // no neighbors means 2 things : current octant is
          // either touching external border
          // either the corner is at hanging face (meaning neighbor is larger)
          // still in both case, we use the same "code"
          if (neigh.size() == 0)
          {
            status |= (NEIGH_NONE << 2*icorner2);
          }

          // 1 neighbor 
          else if (neigh.size() == 1)
          {
            
            // retrieve neighbor octant id
            uint32_t iOct_neigh = neigh[0];

            uint8_t level = mesh.getLevel(iOct);
            uint8_t level_neigh = isghost[0] ?
              mesh.getGhostOctant(iOct_neigh)->getLevel() : // neigh is a MPI ghost
              mesh.getLevel(iOct_neigh);                    // neigh is a regular oct

            // neighbor is larger
            if ( level_neigh < level )
            {
              status |= (NEIGH_IS_LARGER << 2*icorner2);
            }

            // neighbor has same size
            else if ( level_neigh == level )
            {
              status |= (NEIGH_IS_SAME_SIZE << 2*icorner2);
            }

            // neighbor is smaller
            else
            {
              status |= (NEIGH_IS_SMALLER << 2*icorner2);
            }
            
          }
                    
        } // end for icorner

        // 3. edge neighbors - evaluate if really needed
        // TODO
        // TODO
        // TODO

        neigh_level_status_host(iOct) = status;

      }); // end Kokkos parallel_for
      
  } // end fill the mirrored array

  // copy array on device
  Kokkos::deep_copy (m_neigh_level_status, neigh_level_status_host);

} // AMRMetaData<3>::update_neighbor_status

} // namespace dyablo
