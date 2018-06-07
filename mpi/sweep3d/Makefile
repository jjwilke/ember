
CC=mpicc
ifeq ($(SSTMAC_SKELETONIZE),1)
CC=sstcc
endif
CFLAGS=-O3 -std=c99
LIBS=

sweep3d: sweep3d.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm sweep3d

