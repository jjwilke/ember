
CC=mpicc
ifeq ($(SSTMAC_SKELETONIZE),1)
CC=sstcc
endif
CFLAGS=-O0 -g -std=c99
LIBS=

sweep3d: sweep3d.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm sweep3d

