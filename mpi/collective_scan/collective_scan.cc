#include <ember-util.h>
#include <mpi.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <tuple>
#include <vector>

static int warm_up_iters = 50;

auto now() { return std::chrono::high_resolution_clock::now(); }

template <typename Tp>
auto diff_in_s(Tp const& first, Tp const& second) {
  return std::chrono::duration<double>(second - first).count();
}

using return_type = std::tuple<double, double, std::vector<double>>;
class CollectiveRunner {
 public:
  CollectiveRunner(MPI_Comm comm) { MPI_Comm_dup(comm, &comm_); }

  // Returns the average time and a list of times on this node.
  template <typename F, typename... Args>
  std::tuple<double, double, std::vector<double>> run_collective(
      int iters, F&& f, Args&&... args) {
    double total_time = 0;
    double total_time2 = 0;

    std::vector<double> itimes;
    itimes.reserve(iters);

    for (auto i = 0; i < iters + warm_up_iters; ++i) {
      const auto t0 = now();

      if (std::forward<F>(f)(std::forward<Args>(args)..., comm_) !=
          MPI_SUCCESS) {
        std::cerr << "Function failed to return success\n";
      }

      if (i >= warm_up_iters) {
        const auto iter_time = diff_in_s(t0, now());
        itimes.emplace_back(iter_time);
        total_time += iter_time;
        total_time2 += iter_time * iter_time;
      }
    }

    const double average_time = total_time / double(iters);
    const double average_time2 = total_time2 / double(iters);
    const double std_dev =
        std::sqrt(average_time2 - average_time * average_time);

    return {average_time, std_dev, itimes};
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

return_type run_Bcast(int nelements, int niters, CollectiveRunner& crun) {
  auto data = std::vector<int>(nelements);
  std::fill(data.begin(), data.end(), crun.rank());
  return crun.run_collective(niters, MPI_Bcast, data.data(), nelements, MPI_INT,
                             0);
}

return_type run_Allgather(int nelements, int niters, CollectiveRunner& crun) {
  auto data = std::vector<int>(nelements);
  auto out_data = std::vector<int>(crun.size() * nelements);
  std::fill(data.begin(), data.end(), crun.rank());
  return crun.run_collective(niters, MPI_Allgather, data.data(), nelements,
                             MPI_INT, out_data.data(), nelements, MPI_INT);
}

return_type run_Alltoall(int nelements, int niters, CollectiveRunner& crun) {
  const auto size = crun.size();
  auto data = std::vector<int>(size * nelements);
  std::fill(data.begin(), data.end(), crun.rank());
  auto out_data = std::vector<int>(size * nelements);
  return crun.run_collective(niters, MPI_Alltoall, data.data(), nelements,
                             MPI_INT, out_data.data(), nelements, MPI_INT);
}

return_type run_Allreduce(int nelements, int niters, CollectiveRunner& crun) {
  auto data = std::vector<int>(nelements);
  std::fill(data.begin(), data.end(), crun.rank());
  auto out_data = std::vector<int>(nelements);
  return crun.run_collective(niters, MPI_Allreduce, data.data(),
                             out_data.data(), nelements, MPI_INT, MPI_SUM);
}

std::map<std::string, std::function<return_type(int, int, CollectiveRunner&)>>
    collective_function_map = {{"Bcast", run_Bcast},
                               {"Allgather", run_Allgather},
                               {"Alltoall", run_Alltoall},
                               {"Allreduce", run_Allreduce}};

// Took these two guys from
// https://stackoverflow.com/questions/865668/how-to-parse-command-line-arguments-in-c
char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}
bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

int main(int argc, char** argv) {
  if (cmdOptionExists(argv, argv + argc, "-h")) {
    std::cout << "Program options with defaults:\n";
    std::cout << "-iterations : " << 100 << " number of times to collect times for each collective\n";
    std::cout << "-scramble   : " << 0 << " seed to help generate a new proc ordering\n";
    std::cout << "-warmup     : " << warm_up_iters << " number of iterations to run before collecting data" << std::endl;
    return 0;
  }

  MPI_Init(&argc, &argv);


  const std::uint_fast32_t scramble_seed = [&argc, &argv] {
    if (cmdOptionExists(argv, argv + argc, "-scramble")) {
      return std::stoi(getCmdOption(argv, argv + argc, "-scramble"));
    } else {
      return 0;
    }
  }();

  const int nrepeats = [&argc, &argv] {
    if (cmdOptionExists(argv, argv + argc, "-iterations")) {
      return std::stoi(getCmdOption(argv, argv + argc, "-iterations"));
    } else {
      return 100;
    }
  }();

  if (cmdOptionExists(argv, argv + argc, "-warmup")) {
    warm_up_iters = std::stoi(getCmdOption(argv, argv + argc, "-warmup"));
  }

  MPI_Comm my_world_comm = [&scramble_seed] {
    MPI_Comm out_comm = MPI_COMM_NULL;
    if (scramble_seed > 0) {
      generate_scramble(scramble_seed, MPI_COMM_WORLD, &out_comm);
    } else {
      MPI_Comm_dup(MPI_COMM_WORLD, &out_comm);
    }
    return out_comm;
  }();
  print_hostnames("MPI_COMM_WORLD", MPI_COMM_WORLD);
  print_hostnames("Comm used for running jobs", my_world_comm);

  int rank = -1;
  int size = -1;
  MPI_Comm_rank(my_world_comm, &rank);
  MPI_Comm_size(my_world_comm, &size);

  if(rank == 0){
    std::cout << "nrepeats     : " << nrepeats << "\n";
    std::cout << "scramble seed: " << scramble_seed << "\n";
    std::cout << "warmup iters : " << warm_up_iters << "\n" << std::endl;
  }
  MPI_Barrier(my_world_comm);

  const std::vector<int> num_ints_to_send = [] {
    std::vector<int> out = {64, 4096, 65536};  // Bytes
    const auto int_size = sizeof(int);
    for (auto& bs : out) {
      bs /= int_size;
    }
    return out;
  }();


  CollectiveRunner crun(my_world_comm);
  for (auto num_ints : num_ints_to_send) {
    if (rank == 0) {
      std::cout << "Message size: " << num_ints << " ints " << num_ints * sizeof(int) << " bytes" << std::endl;
      printf("\t%-10s: %15s %15s %15s %15s\n", "Name", "Avg Time0", "STD DEV0",
             "Avg TimeG", "STD DEVG");
    }
    MPI_Barrier(my_world_comm);

    for (auto const& f_pair : collective_function_map) {
      auto& name = f_pair.first;
      auto& func = f_pair.second;
      double avg_time = 0;
      double std_dev = 0;
      std::vector<double> all_times;
      std::tie(avg_time, std_dev, all_times) = func(num_ints, nrepeats, crun);

      // Get all  the timeings
      std::vector<double> global_times(nrepeats * size);
      MPI_Gather(all_times.data(), nrepeats, MPI_DOUBLE, global_times.data(),
                 nrepeats, MPI_DOUBLE, 0, my_world_comm);

      double gavg_total = 0;
      double gavg_total2 = 0;
      for (auto const& el : global_times) {
        gavg_total += el;
        gavg_total2 += el * el;
      }
      gavg_total /= global_times.size();
      gavg_total2 /= global_times.size();

      double gstd_dev = std::sqrt(gavg_total2 - gavg_total * gavg_total);

      if (rank == 0) {
        printf("\t%-10s: %15.9f %15.9f %15.9f %15.9f\n", name.c_str(), avg_time,
               std_dev, gavg_total, gstd_dev);
      }
    }
  }

  MPI_Barrier(my_world_comm);
  MPI_Finalize();

  return 0;
}
