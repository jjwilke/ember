
CXX = mpicxx
CXXFLAGS = -O3 -std=c++11 
CPPFLAGS = -I../util
LIBS=
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX = sst++
CXXFLAGS := $(CXXFLAGS) -fPIC
endif

sweep3d: sweep3d.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LIBS)

clean:
	rm sweep3d

