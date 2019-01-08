
CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX=sst++
endif
CXXFLAGS=-O3 -std=c++11
CPPFLAGS=-I../util
LIBS=

scramble: scramble.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm scramble

