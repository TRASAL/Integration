// Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <ctime>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Integration.hpp>


int main(int argc, char *argv[]) {
  unsigned int padding = 0;
  unsigned int integration = 0;
  bool printCode = false;
  bool printResults = false;
  bool random = false;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
  long long unsigned int wrongSamples = 0;
  PulsarSearch::integrationDMsSamplesConf conf;
  AstroData::Observation observation;

  try {
    isa::utils::ArgumentList args(argc, argv);
    printCode = args.getSwitch("-print_code");
    printResults = args.getSwitch("-print_results");
    random = args.getSwitch("-random");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    integration = args.getSwitchArgument< unsigned int >("-integration");
    conf.setNrSamplesPerBlock(args.getSwitchArgument< unsigned int >("-sb"));
		conf.setNrSamplesPerThread(args.getSwitchArgument< unsigned int >("-st"));
		observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0f, 0.0f);
	} catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception & err ) {
    std::cerr << "Usage: " << argv[0] << " [-print_code] [-print_results] [-random] -opencl_platform ... -opencl_device ... -padding ... -integration ... -sb ... -st ... -samples ... -dms ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
  std::vector< dataType > input(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond(padding / sizeof(dataType)));
  std::vector< dataType > output(observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(dataType)));
  std::vector< dataType > output_control(observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(dataType)));
  cl::Buffer input_d;
  cl::Buffer output_d;
  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, input.size() * sizeof(dataType), 0, 0);
    output_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, output.size() * sizeof(dataType), 0, 0);
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(0));
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    if ( printResults ) {
      std::cout << dm << ": ";
    }
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      if ( random ) {
        input[(dm * observation.getNrSamplesPerPaddedSecond(padding / sizeof(dataType))) + sample] = rand() % 10;
      } else {
        input[(dm * observation.getNrSamplesPerPaddedSecond(padding / sizeof(dataType))) + sample] = sample % 10;
      }
      if ( printResults ) {
        std::cout << input[(dm * observation.getNrSamplesPerPaddedSecond(padding / sizeof(dataType))) + sample] << " ";
      }
    }
    if ( printResults ) {
      std::cout << std::endl;
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(dataType), reinterpret_cast< void * >(input.data()), 0, 0);
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	// Generate kernel
  std::string * code = PulsarSearch::getIntegrationDMsSamplesOpenCL< dataType >(conf, observation.getNrSamplesPerSecond(), dataName, integration, padding);
  cl::Kernel * kernel;
  if ( printCode ) {
    std::cout << *code << std::endl;
  }
	try {
    kernel = isa::OpenCL::compile("integrationDMsSamples" + isa::utils::toString(integration), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
	} catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
		return 1;
	}

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global(conf.getNrSamplesPerBlock() * ((observation.getNrSamplesPerSecond() / integration) / conf.getNrSamplesPerThread()), observation.getNrDMs());
    cl::NDRange local(conf.getNrSamplesPerBlock(), 1);

    std::cout << std::endl;
    std::cout << "Global: " << conf.getNrSamplesPerBlock() * ((observation.getNrSamplesPerSecond() / integration) / conf.getNrSamplesPerThread()) << " " << observation.getNrDMs() << std::endl;
    std::cout << "Local: " << conf.getNrSamplesPerBlock() << " " << 1 << std::endl;
    std::cout << std::endl;

    kernel->setArg(0, input_d);
    kernel->setArg(1, output_d);
    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
    PulsarSearch::integrationDMsSamples(observation, integration, padding, input, output_control);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(dataType), reinterpret_cast< void * >(output.data()));
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error kernel execution: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
		for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond() / integration; sample++ ) {
      if ( !isa::utils::same(output_control[(dm * isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(dataType))) + sample], output[(dm * isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(dataType))) + sample]) ) {
        wrongSamples++;
			}
		}
	}
  if ( printResults ) {
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
      std::cout << dm << ": ";
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond() / integration; sample++ ) {
        std::cout << output_control[(dm * isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(dataType))) + sample] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
      std::cout << dm << ": ";
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond() / integration; sample++ ) {
        std::cout << output[(dm * isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(dataType))) + sample] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast< long long unsigned int >(observation.getNrDMs()) * (observation.getNrSamplesPerSecond() / integration)) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

