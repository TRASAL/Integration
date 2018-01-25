cmake_minimum_required(VERSION 3.10)
project(Integration VERSION 2.1)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(TARGET_LINK_LIBRARIES integration isa_utils isa_opencl astrodata OpenCL)
if($ENV{LOFAR})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_HDF5")
  set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} hdf5 hdf5_cpp z)
endif()
if($ENV{PSRDADA})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PSRDADA")
  set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} psrdada cudart)
endif()

# libintegration
add_library(integration SHARED
  src/Integration.cpp
)
set_target_properties(integration PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "include/Integration.hpp"
)
target_include_directories(integration PRIVATE include)

# IntegrationTesting
add_executable(IntegrationTesting
  src/IntegrationTest.cpp
)
target_include_directories(IntegrationTesting PRIVATE include)
target_link_libraries(IntegrationTesting PRIVATE ${TARGET_LINK_LIBRARIES})

# IntegrationTuning
add_executable(IntegrationTuning
  src/IntegrationTuning.cpp
)
target_include_directories(IntegrationTuning PRIVATE include)
target_link_libraries(IntegrationTuning PRIVATE ${TARGET_LINK_LIBRARIES})

install(TARGETS integration IntegrationTesting IntegrationTuning
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
