// Copyright 2009-2018 Sandia Corporation. Under the terms
// of Contract DE-NA0003525 with Sandia Corporation, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2018, Sandia Corporation
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#include <ember-util.h>
#include <mpi.h>
#include <stdlib.h>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define sstmac_app_name scramble_test
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_me = -1;
  int world_size = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_me);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int size = world_size;
  int me = world_me;

  MPI_Comm scramble_comm = MPI_COMM_WORLD;

  // Initial hostnames
  std::array<char, 256> world_host =
      print_hostnames("MPI_COMM_WORLD", MPI_COMM_WORLD);

  bool did_scramble = false;
  std::array<unsigned long, 3> scramble_info;

  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "-scramble") == 0) {
      did_scramble = true;
      int seed = std::stoi(argv[i + 1]);
      scramble_info = generate_scramble(seed, MPI_COMM_WORLD, &scramble_comm);
      MPI_Comm_rank(scramble_comm, &me);
      ++i;
    } else {
      exit(-1);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  std::array<char, 256> scram_host =
      print_hostnames("Scrambled Comm", scramble_comm);

  MPI_Barrier(MPI_COMM_WORLD);

  // Check Gather
  if (world_me == 0) {
    std::vector<char> org_host(256 * world_size);
    std::vector<char> new_host(256 * world_size);
    std::vector<int> new_rank(world_size);
    std::vector<std::array<unsigned long, 3>> all_scramble_info(world_size);

#pragma sst keep
    MPI_Gather(&me, 1, MPI_INT, new_rank.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

#pragma sst keep
    MPI_Gather(world_host.data(), 256, MPI_CHAR, org_host.data(), 256, MPI_CHAR,
               0, MPI_COMM_WORLD);

    // Deal with scram stuff since apparently allgather is broken for me in sst
#pragma sst keep
    MPI_Bcast(&me, 1, MPI_INT, 0, MPI_COMM_WORLD);
#pragma sst keep
    MPI_Gather(scram_host.data(), 256, MPI_CHAR, new_host.data(), 256, MPI_CHAR,
               me, scramble_comm);

#pragma sst keep
    MPI_Gather(scramble_info.data(), 3, MPI_UNSIGNED_LONG,
               all_scramble_info.data(), 3, MPI_UNSIGNED_LONG, 0,
               MPI_COMM_WORLD);

    std::cout
        << "(COMM_WORLD_RANK -> SCRAM_RANK), (COMM_WORLD_HOST -> SCRAM_HOST)\n";
    std::cout << "\t(seed, rank first value, key)\n";
    for (auto i = 0; i < world_size; ++i) {
      auto old_name = std::string(&org_host[256 * i]);
      auto new_name = std::string(&new_host[256 * i]);

      std::cout << "(" << i << " -> " << new_rank[i] << "), (" << old_name
                << " -> " << new_name << ")\n";

      auto key_info = all_scramble_info[i];
      std::cout << "\t(" << key_info[0] << ", " << key_info[1] << ", "
                << key_info[2] << ")\n";
    }
  } else {
#pragma sst keep
    MPI_Gather(&me, 1, MPI_INT, nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
#pragma sst keep
    MPI_Gather(world_host.data(), 256, MPI_CHAR, nullptr, 256, MPI_CHAR, 0,
               MPI_COMM_WORLD);

    // Get new root for scrambled comm
    int scram_root = -1;
#pragma sst keep
    MPI_Bcast(&scram_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
#pragma sst keep
    MPI_Gather(scram_host.data(), 256, MPI_CHAR, nullptr, 256, MPI_CHAR,
               scram_root, scramble_comm);

#pragma sst keep
    MPI_Gather(scramble_info.data(), 3, MPI_UNSIGNED_LONG, nullptr, 3,
               MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
