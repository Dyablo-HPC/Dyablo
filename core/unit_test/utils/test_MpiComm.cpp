#include <boost/test/unit_test.hpp>

#include "utils/mpi/GlobalMpiSession.h"

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_CASE(test_MpiComm_Allgather)
{
  const MpiComm& comm = dyablo::GlobalMpiSession::get_comm_world();

  int mpi_rank = comm.MPI_Comm_rank();
  int mpi_nproc = comm.MPI_Comm_size();

  int rank_size = 5;
  int total_size = rank_size*mpi_nproc;

  std::vector<double> buff_send(rank_size);
  for(int i=0; i<rank_size; i++)
    buff_send[i] = mpi_rank*rank_size + i+1;
  std::vector<double> buff_recv(total_size);

  comm.MPI_Allgather( buff_send.data(), buff_recv.data(), rank_size );
  BOOST_TEST(total_size!=0);
  for(int i=0; i<total_size; i++)
    BOOST_TEST(buff_recv[i] == i+1);
}

BOOST_AUTO_TEST_CASE(test_MpiComm_Allgatherv_inplace)
{
  const MpiComm& comm = dyablo::GlobalMpiSession::get_comm_world();

  int mpi_rank = comm.MPI_Comm_rank();
  int mpi_nproc = comm.MPI_Comm_size();

  auto rank_size = [](int rank) { return rank+1; };

  int size_local = rank_size(mpi_rank);
  int size_tot = 0;
  for(int i=0; i<mpi_nproc; i++)
    size_tot += rank_size(i);
  int begin_local = 0;
  for(int i=0; i<mpi_rank; i++)
    begin_local += rank_size(i);
  std::vector<double> buff(size_tot);
  
  for( int i=begin_local; i<begin_local+size_local; i++ )
    buff[i] = i+1;
  
  comm.MPI_Allgatherv_inplace( buff.data(), size_local );

  for(int i=0; i<size_tot; i++)
    BOOST_TEST( buff[i] == i+1 );
} 

BOOST_AUTO_TEST_SUITE_END() /* dyablo */