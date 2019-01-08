#include <mpi.h>
#include <unistd.h>
#include <array>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

// Make bad, but portable uniform_int_dist
struct uniorm_int_distribution {
  template <typename Gen>
  int operator()(Gen& g) {
    auto range = b + 1 - a;
    auto recompute_size = g.max() - g.max() % range;
    while (true) {
      int value = g();
      if (value < recompute_size) {
        return value % range;
      }
    }
  }

  std::uint_fast32_t a;
  std::uint_fast32_t b;
};

std::array<unsigned long, 4> generate_scramble(std::uint_fast32_t seed,
                                               MPI_Comm incomm,
                                               MPI_Comm* outcomm) {
  int me;
  MPI_Comm_rank(incomm, &me);
  int size;
  MPI_Comm_size(incomm, &size);

  std::seed_seq sseq{1, 2, 7};
  std::mt19937 gen(sseq);
  std::uint_fast32_t intial_gen_val = gen();
  if (seed > 0) {
    for (auto i = 0; i < me * seed; ++i) {
      gen();
    }
  }
  std::uint_fast32_t final_gen_val = gen();

  // std::uniform_int_distribution<int> int_dist(0, size-1);
  uniorm_int_distribution int_dist{0u,
                                   static_cast<std::uint_fast32_t>(size - 1)};
  int color = 0;  // all the same color - just resort
  int key = int_dist(gen);
  MPI_Comm_split(incomm, color, key, outcomm);

  int test_size;
  MPI_Comm_size(*outcomm, &test_size);
  if (test_size != size) {
    std::cerr << "Comm split partitioned - not permuted - communicator"
              << std::endl;
    MPI_Abort(incomm, 1);
  }

  return {{static_cast<unsigned long>(seed),
           static_cast<unsigned long>(intial_gen_val),
           static_cast<unsigned long>(final_gen_val),
           static_cast<unsigned long>(key)}};
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

