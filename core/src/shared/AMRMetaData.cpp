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
void AMRMetaData<2>::update_neigh_level_status(const AMRmesh& mesh)
{

  const uint8_t nbFaces = 2*m_dim;
  
  const uint8_t nbCorners = m_dim == 2 ? 4 : 8;

  // resize to current number of regular octants
  Kokkos::resize(m_neigh_level_status, m_nbOctants);
  Kokkos::resize(m_neigh_rel_pos_status, m_nbOctants+m_nbGhosts);

  // create mirror on host
  neigh_level_status_array_t::HostMirror neigh_level_status_host = 
    Kokkos::create_mirror_view(m_neigh_level_status);

  neigh_rel_pos_status_array_t::HostMirror neigh_rel_pos_status_host = 
    Kokkos::create_mirror_view(m_neigh_rel_pos_status);

  // fill the mirrored array on host with OpenMP exec space
  {
    Kokkos::RangePolicy<Kokkos::OpenMP> policy(0, m_nbOctants);
    Kokkos::parallel_for(
      policy,
      [&](const uint64_t iOct)
      {
        neigh_level_status_t status = 0;
        neigh_rel_pos_status_t status2 = 0;

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

            // neighbor is larger, 
            // we also need to update status2 (relative position)
            if ( level_neigh < level )
            {
              status |= (NEIGH_IS_LARGER << 2*iface);

              // we need to update status2
              auto pos = get_relative_position(mesh,
                                               iOct,
                                               iOct_neigh,
                                               isghost[0],
                                               DIR_ID(iface/2), // direction
                                               FACE_ID(iface%2),  /* face */
                                               NEIGH_IS_LARGER);
              
              // in 2D only 1 bit per face
              status2 |= (pos << iface);

            } // end neighbor is larger

            // neighbor has same size
            else
            {
              status |= (NEIGH_IS_SAME_SIZE << 2*iface);
            }
            
          } // end neigh.size() == 1

          // more neighbors means neighbors are smaller
          else if (neigh.size() > 1)
          {
            status |= (NEIGH_IS_SMALLER << 2*iface);

            // if neighbor is ghost, we need to update status2 in all
            // the neighbor ghost cells
            if (isghost[0])
            {
              
              for (size_t ighost=0; ighost<neigh.size(); ++ighost)
              {

                auto iOct_n = neigh[ighost];

                // get relative position
                auto pos = get_relative_position(mesh,
                                                 iOct,
                                                 iOct_n,
                                                 true, // isghost
                                                 DIR_ID(iface/2), // direction
                                                 FACE_ID(iface%2),  /* face */
                                                 NEIGH_IS_SMALLER);

                // invert face, because a left face from current octant
                // is a right face from neighbor octant
                // since iface = face + 2 * dir, we just need
                // to toggle lest significant bit
                int iface_neigh = (iface ^ 0x1);

                // modify external status in the neighbor,
                // be careful, there might be a data race, if multiple
                // threads want to update the same ghost octant
                // that will happen quite a lot
                Kokkos::atomic_or(&(neigh_rel_pos_status_host(iOct_n)),
                                  static_cast<neigh_rel_pos_status_t>((pos << iface_neigh)) );
              
              } // end for ighost

            } // end if isghost[0] true
            
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
        neigh_rel_pos_status_host(iOct) = status2;

      }); // end Kokkos parallel_for

  } // end fill the mirrored array

  // copy array on device
  Kokkos::deep_copy (m_neigh_level_status, neigh_level_status_host);
  Kokkos::deep_copy (m_neigh_rel_pos_status, neigh_rel_pos_status_host);

} // AMRMetaData<2>::update_neigh_level_status

// =============================================
// =============================================
// template specialization for 3D
template<>
void AMRMetaData<3>::update_neigh_level_status(const AMRmesh& mesh)
{

  //
  // TODO - NOT UP TO DATE - FIX ME !!!
  //

  const uint8_t nbFaces = 2*m_dim;
  
  const uint8_t nbCorners = m_dim == 2 ? 4 : 8;

  // resize to current number of regular octants
  Kokkos::resize(m_neigh_level_status, m_nbOctants);
  Kokkos::resize(m_neigh_rel_pos_status, m_nbOctants+m_nbGhosts);
  
  // create a mirror on host
  neigh_level_status_array_t::HostMirror neigh_level_status_host = 
    Kokkos::create_mirror_view(m_neigh_level_status);

  // fill the mirrored array on host with OpenMP exec space
  {
    Kokkos::RangePolicy<Kokkos::OpenMP> policy(0, m_nbOctants);
    Kokkos::parallel_for(
      policy,
      [&](const uint64_t iOct)
      {

        neigh_level_status_t status = 0;

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

} // AMRMetaData<3>::update_neigh_level_status

// ==============================================================
// ==============================================================
/**
 * Identifies situations like this (assuming neighbor octant 
 * is on the left, and current octant on the right) :
 *
 * \return either NEIGH_POS_0 or NEIGH_POS_1 (see below)
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
 *
 * \param[in] iOct octant id of current current octant
 * \param[in] iOct_neigh octant id of current neighbor octant
 * \param[in] is_ghost true if neighbor is ghost octant
 * \param[in] dir identifies the direction of the interface
 * \param[in] face identifies the face (left or right)
 */
template<>
typename AMRMetaData<2>::NEIGH_POSITION 
AMRMetaData<2>::get_relative_position(const AMRmesh& mesh,
                                      uint32_t iOct,
                                      uint32_t iOct_neigh,
                                      bool     is_ghost,
                                      DIR_ID   dir,
                                      FACE_ID  face,
                                      NEIGH_LEVEL neigh_size) const
{
  
  // default value
  NEIGH_POSITION res = NEIGH_POS_0;
  
  // the following is a bit dirty, because current PABLO does not allow
  // the user to probe logical coordinates (integer), only physical
  // coordinates (double)
  
  /*
   * check if we are dealing with face along X, Y or Z direction
   */
  if (dir == DIR_X)
  {
    
    // get Y coordinates of the lower left corner of current octant
    real_t cur_loc = mesh.getY(iOct);
    
    // get Y coordinates of the lower left corner of neighbor octant
    real_t neigh_loc = is_ghost ? 
      mesh.getYghost(iOct_neigh) : 
      mesh.getY     (iOct_neigh);
    
    if (neigh_size == NEIGH_IS_LARGER  and (neigh_loc < cur_loc) )
      res = NEIGH_POS_1;
    
    if (neigh_size == NEIGH_IS_SMALLER and (neigh_loc > cur_loc) )
      res = NEIGH_POS_1;
    
  }
  
  if (dir == DIR_Y) 
  {
    
    // get X coordinates of the lower left corner of current octant
    real_t cur_loc = mesh.getX(iOct);
    
    // get X coordinates of the lower left corner of neighbor octant
    real_t neigh_loc = is_ghost ? 
      mesh.getXghost(iOct_neigh) : 
      mesh.getX     (iOct_neigh);
    
    if ( neigh_size == NEIGH_IS_LARGER and (neigh_loc < cur_loc) )
      res = NEIGH_POS_1;
    
    if ( neigh_size == NEIGH_IS_SMALLER and (neigh_loc > cur_loc) )
      res = NEIGH_POS_1;
    
  }
  
  return res;
  
} // AMRMetaData<2>::get_relative_position

// ==============================================================
// ==============================================================
template<>
typename AMRMetaData<3>::NEIGH_POSITION 
AMRMetaData<3>::get_relative_position(const AMRmesh& mesh,
                                      uint32_t iOct,
                                      uint32_t iOct_neigh,
                                      bool     is_ghost,
                                      DIR_ID   dir,
                                      FACE_ID  face,
                                      NEIGH_LEVEL neigh_size) const
{

  // default value
  NEIGH_POSITION res = NEIGH_POS_0;

  // TODO

  return res;

} // AMRMetaData<3>::get_relative_position

} // namespace dyablo
