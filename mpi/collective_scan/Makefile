CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX="/projects/wg-sst/ecp_he_sim_only_final_dir/install/sstmac-no-core/bin/sst++"
endif
CXXFLAGS=-O3 -std=c++11
CPPFLAGS=-I../util
LIBS=

collective_scan: collective_scan.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm collective_scan

