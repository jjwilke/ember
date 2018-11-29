CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX=sst++
endif
CXXFLAGS= -O3 -std=c++11
CPPFLAGS= -I../util

all: subcom-a2a

subcom-a2a : subcom-a2a.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	rm -f subcom-a2a$(SUFFIX)
