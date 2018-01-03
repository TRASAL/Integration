cmake_minimum_required(VERSION 3.10)
project(Integration VERSION 2.1)
include(GNUInstallDirs)

add_library(integration SHARED
  src/Integration.cpp
)
set_target_properties(integration PROPERTIES 
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "include/Integration.hpp"
)
target_include_directories(integration PRIVATE include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g0 -march=native -mtune=native")

add_executable(IntegrationTesting
  src/IntegrationTest.cpp
)
target_include_directories(IntegrationTesting PRIVATE include)
target_link_libraries(IntegrationTesting integration isa_utils isa_opencl astrodata OpenCL)

add_executable(IntegrationTuning
  src/IntegrationTuning.cpp
)
target_include_directories(IntegrationTuning PRIVATE include)
target_link_libraries(IntegrationTuning integration isa_utils isa_opencl astrodata OpenCL)

install(TARGETS integration IntegrationTesting IntegrationTuning
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
