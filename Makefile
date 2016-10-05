
ROOT ?= $(HOME)

# https://github.com/isazi/utils
UTILS := $(ROOT)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(ROOT)/src/OpenCL
# https://github.com/isazi/AstroData
ASTRODATA := $(ROOT)/src/AstroData

INCLUDES := -I"include" -I"$(ASTRODATA)/include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -L/usr/local/cuda-6.0/targets/x86_64-linux/lib -lOpenCL

CC := g++

# Dependencies
DEPS := $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o bin/Integration.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 


all: bin/Integration.o bin/IntegrationTest bin/IntegrationTuning bin/printCode

bin/Integration.o: $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/utils.o include/Integration.hpp src/Integration.cpp
	$(CC) -o bin/Integration.o -c src/Integration.cpp $(CL_INCLUDES) $(CFLAGS)

bin/IntegrationTest: $(CL_DEPS) src/IntegrationTest.cpp
	$(CC) -o bin/IntegrationTest src/IntegrationTest.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/IntegrationTuning: $(CL_DEPS) src/IntegrationTuning.cpp
	$(CC) -o bin/IntegrationTuning src/IntegrationTuning.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/printCode: $(DEPS) src/printCode.cpp
	$(CC) -o bin/printCode src/printCode.cpp $(DEPS) $(INCLUDES) $(LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

