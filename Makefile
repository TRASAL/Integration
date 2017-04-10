
SOURCE_ROOT ?= $(HOME)

# https://github.com/isazi/utils
UTILS := $(SOURCE_ROOT)/utils
# https://github.com/isazi/OpenCL
OPENCL := $(SOURCE_ROOT)/OpenCL
# https://github.com/isazi/AstroData
ASTRODATA := $(SOURCE_ROOT)/AstroData

INCLUDES := -I"include" -I"$(ASTRODATA)/include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"

CFLAGS := -std=c++11 -Wall
ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -O3 -g0
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL

CC := g++

# Dependencies
DEPS := $(ASTRODATA)/bin/Observation.o $(ASTRODATA)/bin/Platform.o $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o bin/Integration.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 


all: bin/Integration.o bin/IntegrationTest bin/IntegrationTuning bin/printCode

bin/Integration.o: $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/utils.o include/Integration.hpp src/Integration.cpp
	-@mkdir -p bin
	$(CC) -o bin/Integration.o -c src/Integration.cpp $(CL_INCLUDES) $(CFLAGS)

bin/IntegrationTest: $(CL_DEPS) src/IntegrationTest.cpp
	-@mkdir -p bin
	$(CC) -o bin/IntegrationTest src/IntegrationTest.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/IntegrationTuning: $(CL_DEPS) src/IntegrationTuning.cpp
	-@mkdir -p bin
	$(CC) -o bin/IntegrationTuning src/IntegrationTuning.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/printCode: $(CL_DEPS) src/printCode.cpp
	-@mkdir -p bin
	$(CC) -o bin/printCode src/printCode.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LDFLAGS) $(CFLAGS)

test: bin/IntegrationTest bin/printCode
	echo "Example kernel:"
	./bin/printCode -samples_dms -padding 32 -vector 32 -integration 4 -samples_per_block 4  -samples_per_thread 5 -samples 20 -dms 20
	./bin/IntegrationTest -opencl_platform 0 -opencl_device 0 -samples_dms -vector 32 -padding 32 -integration 4 -samples_per_block 4 -samples_per_thread 5 -samples 20 -dms 20

tune: bin/IntegrationTuning
	./bin/IntegrationTuning -opencl_platform 0 -opencl_device 0 -padding 32 -vector 32 -integration 4 -min_threads 1 -max_threads 200 -max_items 200 -iterations 3 -samples_dms -dms 1024 -samples 1024

clean:
	-@rm bin/*

