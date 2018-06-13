#include <mpi.h>
#include <array>
#include <complex>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <sys/time.h>
#include <time.h>

struct Input {
  int nx = 100;
  int ny = 100;
  int nz = 100;
  int iters = 1;

  int rows = 1;
  int ns_per_el = 1;

  std::array<long long, 3> fwd_fft_cost = {{1, 1, 1}};
  std::array<long long, 3> bwd_fft_cost = {{1, 1, 1}};

  int nproc = 1;
  int rank = 0;
};

std::ostream &operator<<(std::ostream &os, Input const &in) {
  os << "Input:\n";
  os << "\tnx: " << in.nx << "\n";
  os << "\tny: " << in.ny << "\n";
  os << "\tnz: " << in.nz << "\n";

  os << "\titers: " << in.iters << "\n";
  os << "\trows: " << in.rows << "\n";
  os << "\tcols: " << in.nproc / in.rows << "\n";
  os << "\tns_per_el: " << in.ns_per_el << "\n";

  os << "\tfft forward costs: [" << in.fwd_fft_cost[0] / 1e9 << ", "
     << in.fwd_fft_cost[1] / 1e9 << ", " << in.fwd_fft_cost[2] / 1e9 << "]s\n";
  os << "\tfft backwards costs: [" << in.bwd_fft_cost[0] / 1e9 << ", "
     << in.bwd_fft_cost[1] / 1e9 << ", " << in.bwd_fft_cost[2] / 1e9 << "]s\n";

  os << "\tnproc: " << in.nproc;

  return os;
}

struct MessageWrapper {
  MPI_Comm comm;

  std::vector<int> send_counts_fwd;
  std::vector<int> recv_counts_fwd;
  std::vector<int> send_dsp_fwd;
  std::vector<int> recv_dsp_fwd;

  // Backwards messages not needed for rows
  std::vector<int> send_counts_bwd;
  std::vector<int> recv_counts_bwd;
  std::vector<int> send_dsp_bwd;
  std::vector<int> recv_dsp_bwd;

  std::vector<int> group_ranks;

#pragma sst null_type sstmac::vector push_back data
  std::vector<std::complex<double>> send_buf;

#pragma sst null_type sstmac::vector push_back data
  std::vector<std::complex<double>> recv_buf;
};

Input parse_input(int, char **);
void init_input(Input &);

void dummy_compute(long long cost_in_nano_seconds) {
  struct timespec sleepTS;
  sleepTS.tv_sec = 0;
  sleepTS.tv_nsec = cost_in_nano_seconds;

  struct timespec remainTS;

  if (nanosleep(&sleepTS, &remainTS) == EINTR) {
    while (nanosleep(&remainTS, &remainTS) == EINTR)
      ;
  }
}

// Return row and column communicators
std::pair<MessageWrapper, MessageWrapper> setup_message_wrappers(Input const &);

#define sstmac_app_name fft3d
int main(int argc, char **argv) {
  auto input = parse_input(argc, argv);

  MPI_Init(&argc, &argv);

  // init_input makes communicators and sets up our calculation
  init_input(input);

  if (input.rank == 0) {
    std::cout << input << std::endl;
  }

  auto fwd_func = [&input](int pos) {
    return dummy_compute(input.fwd_fft_cost[pos]);
  };
  auto bwd_func = [&input](int pos) {
    return dummy_compute(input.bwd_fft_cost[pos]);
  };

  // TODO Ask Jer if this should be doubles or complexes
  auto a2a_fwd = [](MessageWrapper &mw) {
    // if (MPI_SUCCESS !=
    //     MPI_Alltoallv(mw.send_buf.get(), mw.send_counts_fwd.data(),
    //                   mw.send_dsp_fwd.data(), MPI_DOUBLE, mw.recv_buf.get(),
    //                   mw.recv_counts_fwd.data(), mw.recv_dsp_fwd.data(),
    //                   MPI_DOUBLE, mw.comm)) {
    //   std::cout << "MPI_Alltoallv forward failed with an error." <<
    //   std::endl; throw;
    // }
    auto message_size = 5;
    if (MPI_SUCCESS != MPI_Alltoall(mw.send_buf.data(), message_size,
                                    MPI_DOUBLE, mw.recv_buf.data(),
                                    message_size, MPI_DOUBLE, mw.comm)) {
      std::cout << "MPI_Alltoall forward failed with an error." << std::endl;
      throw;
    }
  };

  // TODO Ask Jer if this should be doubles or complexes
  auto a2a_bwd = [](MessageWrapper &mw) {
    // if (MPI_SUCCESS !=
    //     MPI_Alltoallv(mw.send_buf.get(), mw.send_counts_bwd.data(),
    //                   mw.send_dsp_bwd.data(), MPI_DOUBLE, mw.recv_buf.get(),
    //                   mw.recv_counts_bwd.data(), mw.recv_dsp_bwd.data(),
    //                   MPI_DOUBLE, mw.comm)) {
    //   std::cout << "MPI_Alltoallv backwards failed with error." << std::endl;
    //   throw;
    // }
    auto message_size = 5;
    if (MPI_SUCCESS != MPI_Alltoall(mw.send_buf.data(), message_size,
                                    MPI_DOUBLE, mw.recv_buf.data(),
                                    message_size, MPI_DOUBLE, mw.comm)) {
      std::cout << "MPI_Alltoall forward failed with an error." << std::endl;
      throw;
    }
  };

  // Get our message wrappers
  MessageWrapper row_messages;
  MessageWrapper col_messages;
  auto rc_mess = setup_message_wrappers(input);
  row_messages = std::move(rc_mess.first);
  col_messages = std::move(rc_mess.second);

  for (auto i = 0; i < input.iters; ++i) {
    // Start running fake FFT here.
    fwd_func(0);
    a2a_fwd(col_messages);

    fwd_func(1);
    a2a_fwd(row_messages);

    fwd_func(2);

    MPI_Barrier(MPI_COMM_WORLD);

    // Begin back
    bwd_func(0);
    a2a_fwd(row_messages);

    bwd_func(1);
    a2a_bwd(col_messages);

    bwd_func(2);
  }

  // End
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_free(&row_messages.comm);
  MPI_Comm_free(&col_messages.comm);
  MPI_Finalize();

  return 0;
}

Input parse_input(int argc, char **argv) {
  Input in;

  if (argc > 1) {
    in.nx = std::stoi(argv[1]);
    in.ny = std::stoi(argv[2]);
    in.nz = std::stoi(argv[3]);
    in.iters = std::stoi(argv[4]);

    in.rows = std::stoi(argv[5]);
    in.ns_per_el = std::stoi(argv[6]);

    in.fwd_fft_cost[0] = std::stoi(argv[7]);
    in.fwd_fft_cost[1] = std::stoi(argv[8]);
    in.fwd_fft_cost[2] = std::stoi(argv[9]);

    in.bwd_fft_cost[0] = std::stoi(argv[10]);
    in.bwd_fft_cost[1] = std::stoi(argv[11]);
    in.bwd_fft_cost[2] = std::stoi(argv[12]);
  }

  return in;
}

void init_input(Input &in) {
  MPI_Comm_size(MPI_COMM_WORLD, &in.nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &in.rank);

  long long flattenZ = (in.ny * in.nz) / in.nproc;
  long long cost = in.ns_per_el * in.nx * flattenZ;
  in.fwd_fft_cost[0] *= cost;
  in.fwd_fft_cost[1] *= cost;
  in.fwd_fft_cost[2] *= cost;

  in.bwd_fft_cost[0] *= cost;
  in.bwd_fft_cost[1] *= cost;
  // Dunno why it's times 2. I just copied from sst-elements ember
  in.bwd_fft_cost[2] *= 2 * cost;
}

std::pair<MessageWrapper, MessageWrapper> setup_message_wrappers(
    Input const &in) {
  MessageWrapper rows_mess;
  MessageWrapper cols_mess;

  // Compute npcols
  auto rows = in.rows;
  auto cols = in.nproc / rows;

  auto my_row = in.rank % rows;
  auto my_col = in.rank / rows;

  // Figure out who else is in our groups
  rows_mess.group_ranks.reserve(cols);
  for (auto i = 0; i < cols; ++i) {
    rows_mess.group_ranks.push_back(i * rows + my_row);
  }

  cols_mess.group_ranks.reserve(rows);
  for (auto i = 0; i < rows; ++i) {
    cols_mess.group_ranks.push_back(my_col * rows + i);
  }

  std::vector<int> nx_loc_row(rows);
  std::vector<int> ny_loc_row(rows);
  std::vector<int> ny_loc_col(cols);
  std::vector<int> nz_loc_col(cols);

  // Compute my nx location
  int nx_half = in.nx / 2 + 1;
  int nx_loc = nx_half / rows;
  for (auto i = 0; i < rows; ++i) {
    nx_loc_row[i] = nx_loc;
  }
  for (auto i = 0; i < nx_half % rows; ++i) {
    ++nx_loc_row[i];
  }
  nx_loc = nx_loc_row[my_row];

  // Compute my ny forward location
  int ny_loc = in.ny / rows;
  for (auto i = 0; i < rows; ++i) {
    ny_loc_row[i] = ny_loc;
  }
  for (auto i = 0; i < in.ny % rows; ++i) {
    ++ny_loc_row[i];
  }
  int ny_loc_fwd = ny_loc_row[my_row];

  // Compute my ny backward location
  ny_loc = in.ny / cols;
  for (auto i = 0; i < cols; ++i) {
    ny_loc_col[i] = ny_loc;
  }
  for (auto i = 0; i < in.ny % cols; ++i) {
    ++ny_loc_col[i];
  }
  int ny_loc_bwd = ny_loc_col[my_col];

  // Compute my nz location
  int nz_loc = in.nz / cols;
  for (auto i = 0; i < cols; ++i) {
    nz_loc_col[i] = nz_loc;
  }
  for (auto i = 0; i < in.nz % cols; ++i) {
    ++nz_loc_col[i];
  }
  nz_loc = nz_loc_col[my_col];

  // Allocate message space for our rows
  rows_mess.send_counts_fwd.reserve(cols);
  rows_mess.send_dsp_fwd.reserve(cols);
  rows_mess.recv_counts_fwd.reserve(cols);
  rows_mess.recv_dsp_fwd.reserve(cols);

  int soffset = 0, roffset = 0;
  for (auto i = 0; i < cols; ++i) {
    const int send_block = 2 * nx_loc * ny_loc_col[i] * nz_loc;
    const int recv_block = 2 * nx_loc * ny_loc_bwd * nz_loc_col[i];

    rows_mess.send_counts_fwd.push_back(send_block);
    rows_mess.recv_counts_fwd.push_back(recv_block);

    rows_mess.send_dsp_fwd.push_back(soffset);
    rows_mess.recv_dsp_fwd.push_back(roffset);

    soffset += send_block;
    roffset += recv_block;
  }

  cols_mess.send_counts_fwd.reserve(rows);
  cols_mess.send_dsp_fwd.reserve(rows);
  cols_mess.recv_counts_fwd.reserve(rows);
  cols_mess.recv_dsp_fwd.reserve(rows);

  soffset = roffset = 0;
  for (auto i = 0; i < rows; ++i) {
    int send_block = 2 * nx_loc_row[i] * ny_loc_fwd * nz_loc;
    int recv_block = 2 * nx_loc * ny_loc_row[i] * nz_loc;

    cols_mess.send_counts_fwd.push_back(send_block);
    cols_mess.recv_counts_fwd.push_back(recv_block);
    cols_mess.send_dsp_fwd.push_back(soffset);
    cols_mess.recv_dsp_fwd.push_back(roffset);

    soffset += send_block;
    roffset += recv_block;
  }

  cols_mess.send_counts_bwd.reserve(rows);
  cols_mess.send_dsp_bwd.reserve(rows);
  cols_mess.recv_counts_bwd.reserve(rows);
  cols_mess.recv_dsp_bwd.reserve(rows);

  soffset = roffset = 0;
  for (auto i = 0; i < rows; ++i) {
    int send_block = 2 * nx_loc * ny_loc_row[i] * nz_loc;
    int recv_block = 2 * nx_loc_row[i] * ny_loc_fwd * nz_loc;

    cols_mess.send_counts_bwd.push_back(send_block);
    cols_mess.recv_counts_bwd.push_back(recv_block);
    cols_mess.send_dsp_bwd.push_back(soffset);
    cols_mess.recv_dsp_bwd.push_back(roffset);

    soffset += send_block;
    roffset += recv_block;
  }

  // Now we need to allocate our send and recieve buffers
  const auto size_x = in.nx * ny_loc_fwd * nz_loc;
  const auto size_y = in.ny * in.nx * nz_loc / rows;
  const auto size_z = in.nz * ny_loc_bwd * in.nx / rows;

  const auto max = std::max({size_x, size_y, size_z});

  // cols_mess.send_buf = std::unique_ptr<char[]>(new char[max * COMPLEX]);
  // cols_mess.recv_buf = std::unique_ptr<char[]>(new char[max * COMPLEX]);

  // rows_mess.send_buf = std::unique_ptr<char[]>(new char[max * COMPLEX]);
  // rows_mess.recv_buf = std::unique_ptr<char[]>(new char[max * COMPLEX]);

  cols_mess.send_buf.resize(max);
  cols_mess.recv_buf.resize(max);

  rows_mess.send_buf.resize(max);
  rows_mess.recv_buf.resize(max);

  // Make the group communicators
  MPI_Group group_all;
  MPI_Comm_group(MPI_COMM_WORLD, &group_all);

  MPI_Group row_group;
  MPI_Group_incl(group_all, rows_mess.group_ranks.size(),
                 rows_mess.group_ranks.data(), &row_group);
  MPI_Comm_create(MPI_COMM_WORLD, row_group, &rows_mess.comm);

  MPI_Group col_group;
  MPI_Group_incl(group_all, cols_mess.group_ranks.size(),
                 cols_mess.group_ranks.data(), &col_group);
  MPI_Comm_create(MPI_COMM_WORLD, col_group, &cols_mess.comm);

  return std::make_pair(std::move(rows_mess), std::move(cols_mess));
}
