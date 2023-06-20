#include <string>

#include "DyabloSession.hpp"
#include "utils/config/ConfigMap.h"
#include "DyabloTimeLoop.h"

int main(int argc, char *argv[])
{
  using namespace dyablo;
  DyabloSession mpi_session(argc, argv);

  if( argc < 2 )
  {
    std::cout << "Error : no input file" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  ./test_solver [--kokkos-***=*] input_file.ini" << std::endl;
    return EXIT_FAILURE;
  }

  /*
   * read parameter file and initialize a ConfigMap object
   */
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap = ConfigMap::broadcast_parameters(input_file);

  DyabloTimeLoop simulation( configMap );

  simulation.run();

  return EXIT_SUCCESS;

} // end main
