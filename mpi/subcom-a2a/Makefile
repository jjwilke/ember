CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
#CXX="/projects/wg-sst/ecp_he_sim_only_final_dir/install/sstmac-yes-core/bin/sst++"
#CXX="/projects/wg-sst/ecp_he_sim_only_final_dir/install/sstmac-yes-core-correct-permissions/bin/sst++"
CXX="/projects/wg-sst/ecp_he_sim_only_final_dir/install/sstmac-no-core/bin/sst++"
endif
CXXFLAGS= -O3 -std=c++11
CPPFLAGS= -I../util

all: subcom-a2a

subcom-a2a : subcom-a2a.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	rm -f subcom-a2a$(SUFFIX)
