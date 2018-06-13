CXX=mpicxx
ifeq ($(SSTMAC_SKELETONIZE),1)
CXX=/Users/canlewi/software/install/sst-macro-debug/bin/sst++
endif
OPTS	= -O0 -g -std=c++14
INCS	= 
SUFFIX	= .x

all: fft$(SUFFIX)

fft$(SUFFIX): fft.cc
	$(CXX) $(OPTS) $(INCS) -o $@ $<

clean:
	rm -f fft$(SUFFIX)

