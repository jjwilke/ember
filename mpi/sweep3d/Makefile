
CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX=sst++
endif
CXXFLAGS=-O3 -std=c++11
CPPFLAGS=-I../util
LIBS=

sweep3d: sweep3d.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LIBS)

clean:
	rm sweep3d

