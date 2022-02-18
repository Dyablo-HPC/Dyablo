/**
 * @file main.cpp
 * @date May 25, 2020
 * @author pkestene
 * @note
 *
 * Contributors: pkestene
 *
 * This file is part of the dyablo software project.
 *
 * @copyright © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
 *
 */

#include <Kokkos_Core.hpp>

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include "DyabloSession.hpp"

// initialization function
bool init_function() { return true; }

// entry point
int main(int argc, char *argv[])
{
  dyablo::DyabloSession mpi_session(argc, argv);
 
  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
