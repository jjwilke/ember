CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX=sst++
endif
OPTS	= -O3 
INCS	= 
SUFFIX	= .x

all: subcom-a2a$(SUFFIX) 

subcom-a2a$(SUFFIX) : subcom-a2a.cc
	$(CXX) $(OPTS) $(INCS) -o $@ $<

clean:
	rm -f subcom-a2a$(SUFFIX)
