/*---------------------------------------------------------------------------*\
 *
 *  bitpit
 *
 *  Copyright (C) 2015-2017 OPTIMAD engineering Srl
 *
 *  -------------------------------------------------------------------------
 *  License
 *  This file is part of bitpit.
 *
 *  bitpit is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License v3 (LGPL)
 *  as published by the Free Software Foundation.
 *
 *  bitpit is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with bitpit. If not, see <http://www.gnu.org/licenses/>.
 *
 \*---------------------------------------------------------------------------*/

#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_PABLO.hpp"

#include "shared/kokkos_shared.h"
#include "shared/SimpleVTKIO.h"

#if BITPIT_ENABLE_MPI==1
#include "PABLO_userDataComm.hpp"
#include "PABLO_userDataLB.hpp"
#endif

using namespace bitpit;

using AppData = Kokkos::View<double*>;

// ======================================================================== //
/*!
  \example PABLO_example_00008.cpp

  \brief Parallel 2D adaptive mesh refinement (AMR) with data using PABLO

  Example 00007 is here enriched by an additional feature.

  The load-balance is performed by keep a family of desired level
  together on the same process.

  In particular in this example the families,
  containing elements from the maximum level reached in the quadtree
  to that level minus 3, are kept on the same partition.

  <b>To run</b>: ./PABLO_example_00008 \n

  <b>To see the result visit</b>: <a
  href="http://optimad.github.io/PABLO/">PABLO website</a> \n

*/
// ======================================================================== //

/**
 * Run the example.
 */
void run()
{
  int iter = 0;

  /**<Instantation of a 2D pablo uniform object.*/
  PabloUniform pablo8(2);

  /**<Set NO 2:1 balance for the octree.*/
  int idx = 0;
  pablo8.setBalance(idx,false);

  /**<Refine globally five level and write the octree.*/
  for (iter=1; iter<6; iter++){
    pablo8.adaptGlobalRefine();
  }

  /**<Define a center point and a radius.*/
  double xc, yc;
  xc = yc = 0.5;
  double radius = 0.25;

  /**<Define vectors of data.*/
  uint32_t nocts = pablo8.getNumOctants();
  uint32_t nghosts = pablo8.getNumGhosts();
  //vector<double> oct_data(nocts, 0.0), ghost_data(nghosts, 0.0);
  AppData oct_data("oct_data", nocts);
  AppData ghost_data("ghost_data", nghosts);
  
  /**<Assign a data (distance from center of a circle) to the octants with at least one node inside the circle.*/
  for (unsigned int i=0; i<nocts; i++){
    /**<Compute the nodes of the octant.*/
    vector<array<double,3> > nodes = pablo8.getNodes(i);
    /**<Compute the center of the octant.*/
    array<double,3> center = pablo8.getCenter(i);
    for (int j=0; j<4; j++){
      double x = nodes[j][0];
      double y = nodes[j][1];
      if ((pow((x-xc),2.0)+pow((y-yc),2.0) <= pow(radius,2.0))){
	oct_data(i) = (pow((center[0]-xc),2.0)+pow((center[1]-yc),2.0));
      }
    }
  }

  /**<Update the connectivity and write the octree.*/
  iter = 0;
  pablo8.updateConnectivity();
  {
    euler_pablo::writeTest(pablo8, "pablo00008_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data);
  }

  /**<Adapt two times with data injection on new octants.*/
  int start = 1;
  for (iter=start; iter<start+2; iter++){
    for (unsigned int i=0; i<nocts; i++){
      /**<Compute the nodes of the octant.*/
      vector<array<double,3> > nodes = pablo8.getNodes(i);
      /**<Compute the center of the octant.*/
      array<double,3> center = pablo8.getCenter(i);
      for (int j=0; j<4; j++){
	double x = nodes[j][0];
	double y = nodes[j][1];
	if ((pow((x-xc),2.0)+pow((y-yc),2.0) <= pow(radius,2.0))){
	  if (center[0]<=xc){

	    /**<Set to refine to the octants in the left side of the domain inside a circle.*/
	    pablo8.setMarker(i,1);
	  }
	  else{

	    /**<Set to coarse to the octants in the right side of the domain inside a circle.*/
	    pablo8.setMarker(i,-1);
	  }
	}
      }
    }

    /**<Adapt the octree and map the data in the new octants.*/
    //vector<double> oct_data_new;
    AppData oct_data_new("oct_data_new");
    vector<uint32_t> mapper;
    vector<bool> isghost;
    pablo8.adapt(true);
    nocts = pablo8.getNumOctants();
    Kokkos::resize(oct_data_new,nocts);

    /**<Assign to the new octant the average of the old children if it is new after a coarsening;
     * while assign to the new octant the data of the old father if it is new after a refinement.
     */
    for (uint32_t i=0; i<nocts; i++){
      pablo8.getMapping(i, mapper, isghost);
      if (pablo8.getIsNewC(i)){
	for (int j=0; j<4; j++){
	  if (isghost[j]){
	    oct_data_new(i) += ghost_data(mapper[j])/4;
	  }
	  else{
	    oct_data_new(i) += oct_data(mapper[j])/4;
	  }
	}
      }
      else{
	oct_data_new(i) += oct_data(mapper[0]);
      }
    }

    /**<Update the connectivity and write the octree.*/
    pablo8.updateConnectivity();
    {
      euler_pablo::writeTest(pablo8,"pablo00008_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data_new);
    }

    oct_data = oct_data_new;
  }

#if BITPIT_ENABLE_MPI==1
  /**<PARALLEL TEST: (Load)Balance the octree over the processes with communicating the data.
   * Preserve the family compact up to 4 levels over the max deep reached in the octree.*/
  uint8_t levels = 4;
  UserDataLB<AppData> data_lb(oct_data,ghost_data);
  pablo8.loadBalance(data_lb, levels);
#endif

  /**<Update the connectivity and write the octree.*/
  pablo8.updateConnectivity();
  {
    euler_pablo::writeTest(pablo8,"pablo00008_iter"+to_string(static_cast<unsigned long long>(iter)), data_lb.data);
  }
}

// =================================================================
// =================================================================
/*!
 * Main program.
 */
int main(int argc, char *argv[])
{
  
#if BITPIT_ENABLE_MPI==1
  MPI_Init(&argc,&argv);
#else
  BITPIT_UNUSED(argc);
  BITPIT_UNUSED(argv);
#endif

  int nProcs=1;
  int rank=0;

  Kokkos::initialize(argc, argv);

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";


#if BITPIT_ENABLE_MPI==1
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  }

  // Initialize the logger
  log::manager().initialize(log::SEPARATE, false, nProcs, rank);
  log::cout() << fileVerbosity(log::NORMAL);
  log::cout() << consoleVerbosity(log::QUIET);

  // Run the example
  try {
    run();
  } catch (const std::exception &exception) {
    log::cout() << exception.what();
    //exit(1);
  }

  Kokkos::finalize();
  
#if BITPIT_ENABLE_MPI==1
  MPI_Finalize();
#endif
}
