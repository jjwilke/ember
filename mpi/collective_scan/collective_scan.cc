#include <mpi.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

auto now() { return std::chrono::high_resolution_clock::now(); }

template <typename Tp>
auto diff_in_s(Tp const& first, Tp const& second) {
  return std::chrono::duration<double>(second - first).count();
}

class CollectiveRunner {
 public:
  CollectiveRunner(MPI_Comm comm) { MPI_Comm_dup(comm, &comm_); }

  // Returns the average time and a list of times on this node.
  template <typename F, typename... Args>
  std::pair<double, std::vector<double>> run_collective(int iters,
                                                        bool iter_barrier,
                                                        F&& f, Args&&... args) {
    double total_time = 0;
    std::vector<double> itimes;
    itimes.reserve(iters);

    for (auto i = 0; i < iters; ++i) {
      const auto t0 = now();

      if (std::forward<F>(f)(std::forward<Args>(args)..., comm_) !=
          MPI_SUCCESS) {
        std::cerr << "Function failed to return success\n";
      }

      if (iter_barrier == true) {
        MPI_Barrier(comm_);
      }

      const auto iter_time = diff_in_s(t0, now());
      itimes.emplace_back(diff_in_s(t0, now()));
      total_time += itimes.back();
    }

    return {total_time / double(iters), itimes};
  }

  int rank() const {
    int rank = -1;
    MPI_Comm_rank(comm_, &rank);
    return rank;
  }

  int size() const {
    int size = -1;
    MPI_Comm_size(comm_, &size);
    return size;
  }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;
};

auto run_Bcast(int message_size, CollectiveRunner& crun) {
}

MPI_Comm make_comm(int size, bool randomize, MPI_Comm comm = MPI_COMM_WORLD){
  return comm;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = -1;
  int size = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::vector<int> node_sizes = [] {
    std::vector<int> vec = {16};
    auto i = 0;
    while (vec[i] < 1024) {
      vec.emplace_back(vec.back() + 16);
      ++i;
    }
    vec.back() = 1024;
    return vec;
  }();

  for(auto ns : node_sizes){
    if(ns > size){
       break;
    }

    if(rank == 0){
      std::cout << "Running problem of size: " << ns << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    CollectiveRunner crun(make_comm(ns, false));
    run_Bcast(3000, crun);
  }

  return 0;
}

