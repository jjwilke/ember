#include <mpi.h>
#include <unistd.h>
#include <array>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

int generate_scramble(std::uint_fast32_t seed, MPI_Comm incomm,
                      MPI_Comm* outcomm) {
  int me;
  MPI_Comm_rank(incomm, &me);
  int size;
  MPI_Comm_size(incomm, &size);

  std::seed_seq sseq{1, 2, 7};
  std::mt19937 gen(sseq);
  if (seed > 0) {
    for (auto i = 0; i < me * seed; ++i) {
      gen();
    }
  }

  int color = 0;  // all the same color - just resort
  int key = gen();
  MPI_Comm_split(incomm, color, gen(), outcomm);

  int test_size;
  MPI_Comm_size(*outcomm, &test_size);
  if (test_size != size) {
    std::cerr << "Comm split partitioned - not permuted - communicator"
              << std::endl;
    MPI_Abort(incomm, 1);
  }

  return key;
}

std::array<char, 256> print_hostnames(const char* name, MPI_Comm incomm) {
  char my_host[256];
  int size;
  MPI_Comm_size(incomm, &size);
  int me;
  MPI_Comm_rank(incomm, &me);
  gethostname(my_host, 256);
  int root = 0;
  if (me == root) {
    std::vector<char> all_hosts(256 * size);
    MPI_Gather(my_host, 256, MPI_CHAR, all_hosts.data(), 256, MPI_CHAR, root,
               incomm);
    std::stringstream sstr;
    sstr << name << "\n";
    for (int rank = 0; rank < size; ++rank) {
      const char* name = &all_hosts[256 * rank];
      sstr << "Rank " << rank << ": " << name << "\n";
    }
    std::cout << sstr.str() << std::endl;
  } else {
    MPI_Gather(my_host, 256, MPI_CHAR, nullptr, 256, MPI_CHAR, root, incomm);
  }

  std::array<char, 256> out_array;
  std::copy(my_host, my_host + 256, out_array.begin());
  return out_array;
}

