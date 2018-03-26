
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

void get_position(const int rank, const int pex, const int pey, const int pez, int* myX, int* myY, int* myZ) {

	const int plane  = rank % (pex * pey);
        *myY             = plane / pex;
        *myX             = (plane % pex) != 0 ? (plane % pex) : 0;
        *myZ             = rank / (pex * pey);
}

int convert_position_to_rank(const int pX, const int pY, const int pZ, const int myX, const int myY, const int myZ) {

	// Check if we are out of bounds on the grid
	if(   	(myX < 0) ||
		(myY < 0) ||
		(myZ < 0) ||
		(myX >= pX) ||
		(myY >= pY) ||
		(myZ >= pZ) ) {
		return -1;
	} else {
		return (myZ * (pX * pY)) + (myY * pX) + myX;
	}
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int me = -1;
	int world = -1;

	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &world);

	int pex = world;
	int pey = 1;
	int pez = 1;

	int nx = 10;
	int ny = 10;
	int nz = 10;

	int repeats = 100;

	for(int i = 1; i < argc; i++) {
		if( strcmp(argv[i], "-nx") == 0 ) {
			if( i == argc ) {
				if(me == 0) {
					fprintf(stderr, "Error: specified -nx without a value.\n");
				}

				exit(-1);
			}

			nx = atoi(argv[i+1]);
			++i;
		} else if( strcmp(argv[i], "-ny") == 0 ) {
			if( i == argc ) {
				if(me == 0) {
					fprintf(stderr, "Error: specified -ny without a value.\n");
				}

				exit(-1);
			}

			ny = atoi(argv[i+1]);
			++i;
		} else if( strcmp(argv[i], "-nz") == 0 ) {
			if( i == argc ) {
				if(me == 0) {
					fprintf(stderr, "Error: specified -nz without a value.\n");
				}

				exit(-1);
			}

			nz = atoi(argv[i+1]);
			++i;
		} else if( strcmp(argv[i], "-pex") == 0 ) {
			if( i == argc ) {
				if(me == 0) {
					fprintf(stderr, "Error: specified -pex without a value.\n");
				}

				exit(-1);
			}

			pex = atoi(argv[i+1]);
			++i;
		} else if( strcmp(argv[i], "-pey") == 0 ) {
			if( i == argc ) {
				if(me == 0) {
					fprintf(stderr, "Error: specified -pey without a value.\n");
				}

				exit(-1);
			}

			pey = atoi(argv[i+1]);
			++i;
		} else if( strcmp(argv[i], "-pez") == 0 ) {
			if( i == argc ) {
				if(me == 0) {
					fprintf(stderr, "Error: specified -pez without a value.\n");
				}

				exit(-1);
			}

			pez = atoi(argv[i+1]);
			++i;
		} else if( strcmp(argv[i], "-iterations") == 0 ) {
			if( i == argc ) {
				if(me == 0) {
					fprintf(stderr, "Error: specified -iterations without a value.\n");
				}

				exit(-1);
			}

			repeats = atoi(argv[i+1]);
			++i;
		} else {
			if( 0 == me ) {
				fprintf(stderr, "Unknown option: %s\n", argv[i]);
			}

			exit(-1);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if( (pex * pey * pez) != world ) {
		if( 0 == me) {
			fprintf(stderr, "Error: rank grid does not equal number of ranks.\n");
			fprintf(stderr, "%7d x %7d x %7d != %7d\n", pex, pey, pez, world);
		}

		exit(-1);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if( me == 0 ) {
		printf("# MPI Nearest Neighbor Communication\n");
		printf("# Info:\n");
		printf("# Processor Grid:         %7d x %7d x %7d\n", pex, pey, pez);
		printf("# Data Grid (per rank):   %7d x %7d x %7d\n", nx, ny, nz);
		printf("# Iterations:             %7d\n", repeats);
	}

	int posX, posY, posZ;
	get_position(me, pex, pey, pez, &posX, &posY, &posZ);

	int xUp   = convert_position_to_rank(pex, pey, pez, posX + 1, posY, posZ);
	int xDown = convert_position_to_rank(pex, pey, pez, posX - 1, posY, posZ);
	int yUp   = convert_position_to_rank(pex, pey, pez, posX, posY + 1, posZ);
	int yDown = convert_position_to_rank(pex, pey, pez, posX, posY - 1, posZ);
	int zUp   = convert_position_to_rank(pex, pey, pez, posX, posY, posZ + 1);
	int zDown = convert_position_to_rank(pex, pey, pez, posX, posY, posZ - 1);

	int requestcount = 0;
	MPI_Status* status;
	status = (MPI_Status*) malloc( sizeof(MPI_Status) * 4);

	MPI_Request* requests;
	requests = (MPI_Request*) malloc( sizeof(MPI_Request) * 4);

	double* xUpSendBuffer = (double*) malloc( sizeof(double) * ny * nz );
	double* xUpRecvBuffer = (double*) malloc( sizeof(double) * ny * nz );

	double* xDownSendBuffer = (double*) malloc( sizeof(double) * ny * nz );
	double* xDownRecvBuffer = (double*) malloc( sizeof(double) * ny * nz );

	for(int i = 0; i < ny * nz; i++) {
		xUpSendBuffer[i] = i;
		xUpRecvBuffer[i] = i;
		xDownSendBuffer[i] = i;
		xDownRecvBuffer[i] = i;
	}

	double* yUpSendBuffer = (double*) malloc( sizeof(double) * nx * nz );
	double* yUpRecvBuffer = (double*) malloc( sizeof(double) * nx * nz );

	double* yDownSendBuffer = (double*) malloc( sizeof(double) * nx * nz );
	double* yDownRecvBuffer = (double*) malloc( sizeof(double) * nx * nz );

	for(int i = 0; i < nx * nz; i++) {
		yUpSendBuffer[i] = i;
		yUpRecvBuffer[i] = i;
		yDownSendBuffer[i] = i;
		yDownRecvBuffer[i] = i;
	}

	double* zUpSendBuffer = (double*) malloc( sizeof(double) * nx * ny );
	double* zUpRecvBuffer = (double*) malloc( sizeof(double) * nx * ny );

	double* zDownSendBuffer = (double*) malloc( sizeof(double) * nx * ny );
	double* zDownRecvBuffer = (double*) malloc( sizeof(double) * nx * ny );

	for(int i = 0; i < nx * ny; i++) {
		zUpSendBuffer[i] = i;
		zUpRecvBuffer[i] = i;
		zDownSendBuffer[i] = i;
		zDownRecvBuffer[i] = i;
	}

	struct timeval start;
	struct timeval end;

	gettimeofday( &start, NULL );

	for(int i = 0; i < repeats; ++i) {
		requestcount = 0;

		if( xUp > -1 ) {
			MPI_Irecv(xUpRecvBuffer, ny * nz, MPI_DOUBLE, xUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
			MPI_Isend(xUpSendBuffer, ny * nz, MPI_DOUBLE, xUp, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
		}

		if( xDown > -1 ) {
			MPI_Irecv(xDownRecvBuffer, ny * nz, MPI_DOUBLE, xDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
			MPI_Isend(xDownSendBuffer, ny * nz, MPI_DOUBLE, xDown, 1000, MPI_COMM_WORLD, &requests[requestcount++]);
		}

		MPI_Waitall(requestcount, requests, status);
		requestcount = 0;

		if( yUp > -1 ) {
			MPI_Irecv(yUpRecvBuffer, nx * nz, MPI_DOUBLE, yUp, 2000, MPI_COMM_WORLD, &requests[requestcount++]);
			MPI_Isend(yUpSendBuffer, nx * nz, MPI_DOUBLE, yUp, 2000, MPI_COMM_WORLD, &requests[requestcount++]);
		}

		if( yDown > -1 ) {
			MPI_Irecv(yDownRecvBuffer, nx * nz, MPI_DOUBLE, yDown, 2000, MPI_COMM_WORLD, &requests[requestcount++]);
			MPI_Isend(yDownSendBuffer, nx * nz, MPI_DOUBLE, yDown, 2000, MPI_COMM_WORLD, &requests[requestcount++]);
		}

		MPI_Waitall(requestcount, requests, status);
		requestcount = 0;

		if( zUp > -1 ) {
			MPI_Irecv(zUpRecvBuffer, nx * ny, MPI_DOUBLE, zUp, 4000, MPI_COMM_WORLD, &requests[requestcount++]);
			MPI_Isend(zUpSendBuffer, nx * ny, MPI_DOUBLE, zUp, 4000, MPI_COMM_WORLD, &requests[requestcount++]);
		}

		if( zDown > -1 ) {
			MPI_Irecv(zDownRecvBuffer, nx * ny, MPI_DOUBLE, zDown, 4000, MPI_COMM_WORLD, &requests[requestcount++]);
			MPI_Isend(zDownSendBuffer, nx * ny, MPI_DOUBLE, zDown, 4000, MPI_COMM_WORLD, &requests[requestcount++]);
		}

		MPI_Waitall(requestcount, requests, status);
		requestcount = 0;
	}

	gettimeofday(&end, NULL);

	MPI_Barrier(MPI_COMM_WORLD);

	free(xUpRecvBuffer);
	free(xDownRecvBuffer);
	free(yUpRecvBuffer);
	free(yDownRecvBuffer);
	free(zUpRecvBuffer);
	free(zDownRecvBuffer);

	if( convert_position_to_rank(pex, pey, pez, pex/2, pey/2, pez/2) == me ) {
		printf("# Results from rank: %d\n", me);

		const double timeTaken = ( ((double) end.tv_sec) + ((double) end.tv_usec) * 1.0e-6 ) -
			( ((double) start.tv_sec) + ((double) start.tv_usec) * 1.0e-6 );
		const double bytesXchng =
			((double)( xUp > -1 ? sizeof(double) * ny * nz * 2: 0 )) +
			((double)( xDown > -1 ? sizeof(double) * ny * nz * 2: 0 )) +
			((double)( yUp > -1 ? sizeof(double) * nx * nz * 2: 0 )) +
			((double)( yDown > -1 ? sizeof(double) * nx * nz * 2: 0 )) +
			((double)( zUp > -1 ? sizeof(double) * nx * ny * 2: 0 )) +
			((double)( zDown > -1 ? sizeof(double) * nx * ny * 2: 0 ));

		printf("# %20s %20s %20s\n", "Time", "KBytesXchng/PE-Max", "MB/S");
		printf("  %20.6f %20.4f %20.4f\n",
			timeTaken, bytesXchng / 1024.0, (bytesXchng / 1024.0) / timeTaken );
	}

	MPI_Finalize();
}