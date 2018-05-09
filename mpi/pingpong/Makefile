CC=mpicc
ifeq ($(SSTMAC_SKELETONIZE),1)
CC=sstcc
endif
CFLAGS=-O3 -std=c99
LIBS=

pingpong: pingpong.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm pingpong

