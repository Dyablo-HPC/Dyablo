#include <string>

#include "DyabloSession.hpp"
#include "utils/config/ConfigMap.h"
#include "DyabloTimeLoop.h"

int main(int argc, char *argv[])
{
  using namespace dyablo;
  shared::DyabloSession mpi_session(argc, argv);

  /*
   * read parameter file and initialize a ConfigMap object
   */
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap = ConfigMap::broadcast_parameters(input_file);

  muscl_block::DyabloTimeLoop simulation( configMap );

  simulation.run();

  return EXIT_SUCCESS;

} // end main
