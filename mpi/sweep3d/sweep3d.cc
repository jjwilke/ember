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

#include <errno.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <ember-util.h>

void get_position(const int rank, const int pex, const int pey, int* myX,
                  int* myY) {
  *myX = rank % pex;
  *myY = rank / pex;
}

void compute(long sleep) {
  struct timespec sleepTS;
  sleepTS.tv_sec = 0;
  sleepTS.tv_nsec = sleep;

  struct timespec remainTS;

  if (nanosleep(&sleepTS, &remainTS) == EINTR) {
    while (nanosleep(&remainTS, &remainTS) == EINTR)
      ;
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_me = -1;
  int world_size = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_me);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int me = world_me;
  int size = world_size;
  MPI_Comm sweep_comm = MPI_COMM_WORLD;

  int pex = -1;
  int pey = -1;
  int nx = 50;
  int ny = 50;
  int nz = 100;
  int kba = 10;
  int repeats = 1;

  int vars = 1;
  long sleep = 1000;
  int print = 0;

  for (int i = 0; i < argc; ++i) {
    if (strcmp("-pex", argv[i]) == 0) {
      pex = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-pey", argv[i]) == 0) {
      pey = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-iterations", argv[i]) == 0) {
      repeats = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-nx", argv[i]) == 0) {
      nx = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-ny", argv[i]) == 0) {
      ny = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-nz", argv[i]) == 0) {
      nz = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-sleep", argv[i]) == 0) {
      sleep = atol(argv[i + 1]);
      i++;
    } else if (strcmp("-vars", argv[i]) == 0) {
      vars = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-kba", argv[i]) == 0) {
      kba = atoi(argv[i + 1]);
      i++;
    }  else if (strcmp(argv[i], "-scramble") == 0){
      int scrambler = atol(argv[i+1]);
      generate_scramble(scrambler, MPI_COMM_WORLD, &sweep_comm);
      MPI_Comm_rank(sweep_comm, &me);
      ++i; 
    }  else if (strcmp(argv[i], "-print") == 0){
      print = atoi(argv[i + 1]);
      i++;
    }
  }

  //print_hostnames("MPI_COMM_WORLD", MPI_COMM_WORLD);
  //print_hostnames("Sweep Communicator", sweep_comm);

  if (kba == 0) {
    if (me == 0) {
      fprintf(stderr,
              "K-Blocking Factor must not be zero. Please specify -kba <value "
              "> 0>\n");
    }
    MPI_Barrier(MPI_COMM_WORLD); //needed to force correct printing
    exit(-1);
  }

  if (nz % kba != 0) {
    if (me == 0) {
      fprintf(stderr,
              "KBA must evenly divide NZ, KBA=%d, NZ=%d, remainder=%d (must be "
              "zero)\n",
              kba, nz, (nz % kba));
    }
    MPI_Barrier(MPI_COMM_WORLD); //needed to force correct printing
    exit(-1);
  }

  if ((pex * pey) != size) {
    if (0 == me) {
      fprintf(
          stderr,
          "Error: processor decomposition (%d x %d) != number of ranks (%d)\n",
          pex, pey, size);
    }
    MPI_Barrier(MPI_COMM_WORLD); //needed to force correct printing
    exit(-1);
  }

  if (me == 0 && print) {
    printf("# Sweep3D Communication Pattern\n");
    printf("# Info:\n");
    printf("# Px:              %8d\n", pex);
    printf("# Py:              %8d\n", pey);
    printf("# Nx x Ny x Nz:    %8d x %8d x %8d\n", nx, ny, nz);
    printf("# KBA:             %8d\n", kba);
    printf("# Variables:       %8d\n", vars);
    printf("# Iterations:      %8d\n", repeats);
  }

  int myX = -1;
  int myY = -1;

  get_position(me, pex, pey, &myX, &myY);

  const int xUp = (myX != (pex - 1)) ? me + 1 : -1;
  const int xDown = (myX != 0) ? me - 1 : -1;

  const int yUp = (myY != (pey - 1)) ? me + pex : -1;
  const int yDown = (myY != 0) ? me - pex : -1;

  MPI_Status status;

#pragma sst start_null_variable replace nullptr
  double* xRecvBuffer = (double*)malloc(sizeof(double) * nx * kba * vars);
  double* xSendBuffer = (double*)malloc(sizeof(double) * nx * kba * vars);

  double* yRecvBuffer = (double*)malloc(sizeof(double) * ny * kba * vars);
  double* ySendBuffer = (double*)malloc(sizeof(double) * ny * kba * vars);
#pragma sst stop_null_variable

#pragma sst compute
  for (int i = 0; i < nx; ++i) {
    xRecvBuffer[i] = 0;
    xSendBuffer[i] = i;
  }

#pragma sst compute
  for (int i = 0; i < ny; ++i) {
    yRecvBuffer[i] = 0;
    ySendBuffer[i] = i;
  }

  struct timeval start;
  struct timeval end;

  gettimeofday(&start, NULL);

  // We repeat this sequence twice because there are really 8 vertices in the 3D
  // data domain and we sweep from each of them, processing the top four first
  // and then the bottom four vertices next.
  for (int i = 0; i < (repeats * 2); ++i) {
    // Recreate communication pattern of sweep from (0,0) towards (Px,Py)
    struct timeval iter_start;
    struct timeval iter_end;
    gettimeofday(&iter_start, NULL);
    for (int k = 0; k < nz; k += kba) {
      if (xDown > -1) {
        MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_DOUBLE, xDown, 1000,
                 sweep_comm, &status);
      }

      if (yDown > -1) {
        MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_DOUBLE, yDown, 1000,
                 sweep_comm, &status);
      }

      compute(sleep);

      if (xUp > -1) {
        MPI_Send(xSendBuffer, (nx * kba * vars), MPI_DOUBLE, xUp, 1000,
                 sweep_comm);
      }

      if (yUp > -1) {
        MPI_Send(ySendBuffer, (nx * kba * vars), MPI_DOUBLE, yUp, 1000,
                 sweep_comm);
      }
    }

    // Recreate communication pattern of sweep from (Px,0) towards (0,Py)
    for (int k = 0; k < nz; k += kba) {
      if (xUp > -1) {
        MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_DOUBLE, xUp, 2000,
                 sweep_comm, &status);
      }

      if (yDown > -1) {
        MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_DOUBLE, yDown, 2000,
                 sweep_comm, &status);
      }

      compute(sleep);

      if (xDown > -1) {
        MPI_Send(xSendBuffer, (nx * kba * vars), MPI_DOUBLE, xDown, 2000,
                 sweep_comm);
      }

      if (yUp > -1) {
        MPI_Send(ySendBuffer, (nx * kba * vars), MPI_DOUBLE, yUp, 2000,
                 sweep_comm);
      }
    }

    // Recreate communication pattern of sweep from (Px,Py) towards (0,0)
    for (int k = 0; k < nz; k += kba) {
      if (xUp > -1) {
        MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_DOUBLE, xUp, 3000,
                 sweep_comm, &status);
      }

      if (yUp > -1) {
        MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_DOUBLE, yUp, 3000,
                 sweep_comm, &status);
      }

      compute(sleep);

      if (xDown > -1) {
        MPI_Send(xSendBuffer, (nx * kba * vars), MPI_DOUBLE, xDown, 3000,
                 sweep_comm);
      }

      if (yDown > -1) {
        MPI_Send(ySendBuffer, (nx * kba * vars), MPI_DOUBLE, yDown, 3000,
                 sweep_comm);
      }
    }

    // Recreate communication pattern of sweep from (0,Py) towards (Px,0)
    for (int k = 0; k < nz; k += kba) {
      if (xDown > -1) {
        MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_DOUBLE, xDown, 4000,
                 sweep_comm, &status);
      }

      if (yUp > -1) {
        MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_DOUBLE, yUp, 4000,
                 sweep_comm, &status);
      }

      compute(sleep);

      if (xUp > -1) {
        MPI_Send(xSendBuffer, (nx * kba * vars), MPI_DOUBLE, xUp, 4000,
                 sweep_comm);
      }

      if (yDown > -1) {
        MPI_Send(ySendBuffer, (nx * kba * vars), MPI_DOUBLE, yDown, 4000,
                 sweep_comm);
      }
    }
    gettimeofday(&iter_end, NULL);
    const double timeTaken = (iter_end.tv_sec-iter_start.tv_sec) + (iter_end.tv_usec-iter_start.tv_usec)*1e-6;
    if (print){
      printf("Rank %d = [%d,%d] iteration %d: %12.8fs\n", me, myX, myY, i, timeTaken);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&end, NULL);

  const double timeTaken =
      (((double)end.tv_sec) + ((double)end.tv_usec) * 1.0e-6) -
      (((double)start.tv_sec) + ((double)start.tv_usec) * 1.0e-6);
  const double bytesXchng =
      ((double)repeats) *
      (((double)(xUp > -1 ? sizeof(double) * nx * kba * vars * 2 : 0)) +
       ((double)(xDown > -1 ? sizeof(double) * nx * kba * vars * 2 : 0)) +
       ((double)(yUp > -1 ? sizeof(double) * ny * kba * vars * 2 : 0)) +
       ((double)(yDown > -1 ? sizeof(double) * ny * kba * vars * 2 : 0)));

  if ((myX == (pex / 2)) && (myY == (pey / 2))) {
    if (print){
      printf("# Results from rank: %d\n", me);
      printf("# %20s %20s %20s\n", "Time", "KBytesXchng/Rank-Max", "MB/S/Rank");
      printf("  %20.6f %20.4f %20.4f\n", timeTaken, bytesXchng / 1024.0,
             (bytesXchng / 1024.0) / timeTaken);
    }
  }
  MPI_Finalize();
  return 0;
}
