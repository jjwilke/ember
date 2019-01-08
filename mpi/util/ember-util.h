#include <random>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <mpi.h>
#include <array>

std::array<int, 3> generate_scramble(int seed, MPI_Comm incomm, MPI_Comm* outcomm)
{
  int me; MPI_Comm_rank(incomm, &me);
  int size; MPI_Comm_size(incomm, &size);
  int my_seed = (me + 7) * seed;
  std::mt19937 gen(my_seed);
  std::uniform_int_distribution<int> int_dist(0, size*2);
  int color = 0; //all the same color - just resort
  int key = int_dist(gen);
  MPI_Comm_split(incomm, color, key, outcomm);

  int test_size; MPI_Comm_size(*outcomm, &test_size);
  if (test_size != size){
    std::cerr << "Comm split partitioned - not permuted - communicator" << std::endl;
    MPI_Abort(incomm, 1);
  }
  return {seed, my_seed, key};
}

std::array<char, 256> print_hostnames(const char* name, MPI_Comm incomm)
{
  char my_host[256];
  int size; MPI_Comm_size(incomm, &size);
  int me; MPI_Comm_rank(incomm, &me);
  gethostname(my_host, 256);
  int root = 0;
  if (me == root){
    std::vector<char> all_hosts(256*size);
    MPI_Gather(my_host, 256, MPI_CHAR, 
               all_hosts.data(), 256, MPI_CHAR, 
               root, incomm);
    std::stringstream sstr;
    sstr << name << "\n";
    for (int rank=0; rank < size; ++rank){
      const char* name = &all_hosts[256*rank];
      sstr << "Rank " << rank << ": " << name << "\n";
    }
    std::cout << sstr.str() << std::endl;
  } else {
    MPI_Gather(my_host, 256, MPI_CHAR, nullptr, 256, MPI_CHAR,
               root, incomm);
  }

  std::array<char, 256> out_array;
  std::copy(my_host, my_host + 256, out_array.begin());
  return out_array;
}

