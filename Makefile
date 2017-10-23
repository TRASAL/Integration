
INSTALL_ROOT ?= $(HOME)
INCLUDES := -I"include" -I"$(INSTALL_ROOT)/include"
LIBS := -L"$(INSTALL_ROOT)/lib"

CC := g++
CFLAGS := -std=c++11 -Wall
LDFLAGS := -lm -lOpenCL -lutils -lisaOpenCL -lAstroData

ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -O3 -g0
endif


all: bin/Integration.o bin/IntegrationTest bin/IntegrationTuning
	-@mkdir -p lib
	$(CC) -o lib/libIntegration.so -shared -Wl,-soname,libIntegration.so bin/Integration.o $(CFLAGS)

bin/Integration.o: include/Integration.hpp src/Integration.cpp
	-@mkdir -p bin
	$(CC) -o bin/Integration.o -c -fpic src/Integration.cpp $(INCLUDES) $(CFLAGS)

bin/IntegrationTest: src/IntegrationTest.cpp
	-@mkdir -p bin
	$(CC) -o bin/IntegrationTest src/IntegrationTest.cpp bin/Integration.o $(INCLUDES) $(LIBS) $(LDFLAGS) $(CFLAGS)

bin/IntegrationTuning: src/IntegrationTuning.cpp
	-@mkdir -p bin
	$(CC) -o bin/IntegrationTuning src/IntegrationTuning.cpp bin/Integration.o $(INCLUDES) $(LIBS) $(LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*
	-@rm lib/*

install: all
	-@mkdir -p $(INSTALL_ROOT)/include
	-@cp include/Integration.hpp $(INSTALL_ROOT)/include
	-@mkdir -p $(INSTALL_ROOT)/lib
	-@cp lib/* $(INSTALL_ROOT)/lib
	-@mkdir -p $(INSTALL_ROOT)/bin
	-@cp bin/IntegrationTest $(INSTALL_ROOT)/bin
	-@cp bin/IntegrationTuning $(INSTALL_ROOT)/bin
