#include <stdio.h>
#include <stdlib.h>
#include <ember-util.h>
#include <cstring>
#include "mpi.h"

#define MP_X 0
#define MP_Y 1
#define MP_Z 2

#if WRITE_OTF2_TRACE
#include <scorep/SCOREP_User.h>
#endif

#define calc_pe(a,b,c)  ((a)+(b)*dims[MP_X]+(c)*dims[MP_X]*dims[MP_Y])

int main(int argc, char **argv)
{
  int world_rank, numranks;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numranks);

  int myrank = world_rank;
  MPI_Comm comm = MPI_COMM_WORLD;

  int dims[3] = {0, 0, 0};
  dims[MP_X] = atoi(argv[1]); // nx
  dims[MP_Y] = atoi(argv[2]); // ny
  dims[MP_Z] = atoi(argv[3]); // nz

  int msg_size_x = 0;
  int msg_size_y = 0;
  int msg_size_z = 0;
  int MAX_ITER = 10;
  int print = 0;

  for (int i = 0; i < argc; ++i) {
    if (strcmp("-pex", argv[i]) == 0) {
      dims[MP_X] = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-pey", argv[i]) == 0) {
      dims[MP_Y] = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-pez", argv[i]) == 0) {
      dims[MP_Z] = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-iterations", argv[i]) == 0) {
      MAX_ITER = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-nx", argv[i]) == 0) {
      msg_size_x = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-ny", argv[i]) == 0) {
      msg_size_y = atoi(argv[i + 1]);
      i++;
    } else if (strcmp("-nz", argv[i]) == 0) {
      msg_size_z = atoi(argv[i + 1]);
      i++;
    }  else if (strcmp(argv[i], "-scramble") == 0){
      int scrambler = atol(argv[i+1]);
      generate_scramble(scrambler, MPI_COMM_WORLD, &comm);
      MPI_Comm_rank(comm, &myrank);
      ++i; 
    }  else if (strcmp(argv[i], "-print") == 0){
      print = atol(argv[i+1]);
      ++i;
    }
  }

  //print_hostnames("MPI_COMM_WORLD", MPI_COMM_WORLD);
  //print_hostnames("Subcom Communicator", comm);
  
  if(dims[MP_X] * dims[MP_Y] * dims[MP_Z] != numranks) {
    fprintf(stderr, "\n nx * ny * nz does not equal number of ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  //figure out my coordinates
  int myXcoord = myrank % dims[MP_X];
  int myYcoord = (myrank % (dims[MP_X] * dims[MP_Y])) / dims[MP_X];
  int myZcoord = (myrank % (dims[MP_X] * dims[MP_Y] * dims[MP_Z])) / (dims[MP_X] * dims[MP_Y]);

  bool skip[3];

  //which a2as to skip
  skip[MP_X] = msg_size_x == 0;
  skip[MP_Y] = msg_size_y == 0;
  skip[MP_Z] = msg_size_z == 0;

  //all a2a share the buffer
  int largestMsg = (msg_size_x * dims[MP_X] > msg_size_y * dims[MP_Y]) ? msg_size_x * dims[MP_X] : msg_size_y * dims[MP_Y];
  largestMsg = (largestMsg > msg_size_z * dims[MP_Z]) ? largestMsg : msg_size_z * dims[MP_Z];

#pragma sst null_variable replace nullptr
  char *sendbuf = (char*)malloc(largestMsg);
#pragma sst null_variable replace nullptr
  char *recvbuf = (char*)malloc(largestMsg);

  //create subcommunicators
  MPI_Comm X_comm, Y_comm, Z_comm;
  if(!skip[MP_X]) {
    MPI_Comm_split(comm, myZcoord * dims[MP_Y] + myYcoord, myXcoord, &X_comm);
  }
  if(!skip[MP_Y]) {
    MPI_Comm_split(comm, myZcoord * dims[MP_X] + myXcoord, myYcoord, &Y_comm);
  }
  if(!skip[MP_Z]) {
    MPI_Comm_split(comm, myYcoord * dims[MP_X] + myXcoord, myZcoord, &Z_comm);
  }

  double startTime, stopTime;
  MPI_Barrier(MPI_COMM_WORLD);

  startTime = MPI_Wtime();
  for (int i = 0; i < MAX_ITER; i++) {
    double start = MPI_Wtime();
    if(!skip[MP_X]) {
      MPI_Alltoall(sendbuf, msg_size_x, MPI_CHAR, recvbuf, msg_size_x, MPI_CHAR, X_comm);
    }

    if(!skip[MP_Y]) {
      MPI_Alltoall(sendbuf, msg_size_y, MPI_CHAR, recvbuf, msg_size_y, MPI_CHAR, Y_comm);
    }

    if(!skip[MP_Z]) {
      MPI_Alltoall(sendbuf, msg_size_z, MPI_CHAR, recvbuf, msg_size_z, MPI_CHAR, Z_comm);
    }

    double stop = MPI_Wtime();
    if (print){
      printf("Rank %d = [%d,%d,%d] iteration %d: %12.8fs\n", 
             myrank, myXcoord, myYcoord, myZcoord, i, (stop-start));
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  stopTime = MPI_Wtime();


  //finalized summary output
  if(myrank == 0 && MAX_ITER != 0 && print) {
    printf("Finished %d iterations\n",MAX_ITER);
    printf("Time elapsed per iteration for grid size (%d,%d,%d) with message sizes (%d,%d,%d) : %f s\n", 
    dims[MP_X], dims[MP_Y], dims[MP_Z], msg_size_x, msg_size_y, msg_size_z, (stopTime - startTime)/MAX_ITER);
  }

  MPI_Finalize();
  return 0;
}


