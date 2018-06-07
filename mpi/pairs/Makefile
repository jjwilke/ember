CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX=sst++
endif
OPTS	= -O3 
INCS	= 
SUFFIX	= .x

all: pairs$(SUFFIX)

pairs$(SUFFIX): pairs.cc
	$(CXX) $(OPTS) $(INCS) -o $@ $<

clean:
	rm -f pairs$(SUFFIX)

