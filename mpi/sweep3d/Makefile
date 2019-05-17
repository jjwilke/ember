
CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX="/projects/wg-sst/ecp_he_sim_only_final_dir/install/sstmac-yes-core/bin/sst++"
endif
CXXFLAGS=-O3 -std=c++11
CPPFLAGS=-I../util
LIBS=

sweep3d: sweep3d.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LIBS)

clean:
	rm sweep3d

