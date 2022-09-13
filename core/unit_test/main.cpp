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

#include "gtest/gtest.h"

#include "DyabloSession.hpp"

// entry point
int main(int argc, char *argv[])
{
  dyablo::DyabloSession mpi_session(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
