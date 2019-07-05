
CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX=sst++
endif
CXXFLAGS=-O3 -std=c++11 -fPIC
CPPFLAGS=-I../util
LIBS=

halo3d-26: halo3d-26.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm halo3d-26

