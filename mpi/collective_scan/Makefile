
CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX=/Users/canlewi/software/sst-macro/install/fast_build/bin/sst++
endif
CXXFLAGS=-O3 -std=c++14
CPPFLAGS=-I../util
LIBS=

collective_scan: collective_scan.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm scramble

