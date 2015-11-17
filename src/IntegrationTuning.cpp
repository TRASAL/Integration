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
#include <iomanip>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Integration.hpp>
#include <Timer.hpp>
#include <Stats.hpp>


void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, cl::Buffer * input_d, const unsigned int input_size, cl::Buffer * output_d, const unsigned int output_size);

int main(int argc, char * argv[]) {
  bool reInit = false;
  unsigned int padding = 0;
  unsigned int integration = 0;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int minThreads = 0;
  unsigned int maxThreads = 0;
  unsigned int maxItems = 0;
  unsigned int vectorWidth = 0;
  AstroData::Observation observation;
  PulsarSearch::integrationSamplesDMsConf conf;
  cl::Event event;

	try {
    isa::utils::ArgumentList args(argc, argv);

		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		padding = args.getSwitchArgument< unsigned int >("-padding");
		integration = args.getSwitchArgument< unsigned int >("-integration");
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
		maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    vectorWidth = args.getSwitchArgument< unsigned int >("-vector");
		observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0f, 0.0f);
	} catch ( isa::utils::EmptyCommandLine & err ) {
		std::cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -integration ... -min_threads ... -max_threads ... -max_items ... -vector ... -samples ... -dms ... " << std::endl;
		return 1;
	} catch ( std::exception & err ) {
		std::cerr << err.what() << std::endl;
		return 1;
	}

  // Allocate host memory
  std::vector< dataType > input(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond(padding / sizeof(dataType)));
  std::vector< dataType > output(observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(dataType)));

	// Initialize OpenCL
	cl::Context clContext;
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();
  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);

	// Allocate device memory
  cl::Buffer input_d;
  cl::Buffer output_d;

  try {
    initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &input_d, input.size(), &output_d, output.size());
  } catch ( cl::Error & err ) {
    std::cerr << err.what() << std::endl;
    return -1;
  }

	std::cout << std::fixed << std::endl;
	std::cout << "# nrDMs nrSamples integration samplesPerBlock samplesPerThread GFLOP/s GB/s time stdDeviation COV" << std::endl << std::endl;

	for ( unsigned int samples = minThreads; samples <= maxThreads; samples++) {
    conf.setNrSamplesPerBlock(samples);
    if ( conf.getNrSamplesPerBlock() % vectorWidth != 0 ) {
      continue;
    }

    for ( unsigned int samplesPerThread = 1; samplesPerThread <= maxItems; samplesPerThread++ ) {
      conf.setNrSamplesPerThread(samplesPerThread);
      if ( (observation.getNrSamplesPerSecond() % (integration * conf.getNrSamplesPerThread())) != 0 ) {
        continue;
      }

      // Generate kernel
      double gflops = isa::utils::giga(static_cast< uint64_t >(observation.getNrDMs()) * observation.getNrSamplesPerSecond());
      double gbs = isa::utils::giga((static_cast< uint64_t >(observation.getNrDMs()) * observation.getNrSamplesPerSecond()) + (static_cast< uint64_t >(observation.getNrDMs()) * (observation.getNrSamplesPerSecond() / integration)));
      isa::utils::Timer timer;
      cl::Kernel * kernel;

      std::string * code = PulsarSearch::getIntegrationSamplesDMsOpenCL< dataType >(conf, observation, dataName, integration, padding);
      if ( reInit ) {
        delete clQueues;
        clQueues = new std::vector< std::vector < cl::CommandQueue > >();
        isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
        try {
          initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &input_d, input.size(), &output_d, output.size());
        } catch ( cl::Error & err ) {
          std::cerr << "Error in memory allocation: ";
          std::cerr << isa::utils::toString(err.err()) << "." << std::endl;
          return -1;
        }
        reInit = false;
      }
      try {
        kernel = isa::OpenCL::compile("integrationSamplesDMs" + isa::utils::toString(integration), *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
      } catch ( isa::OpenCL::OpenCLError & err ) {
        std::cerr << err.what() << std::endl;
        delete code;
        break;
      }
      delete code;

      cl::NDRange global((observation.getNrSamplesPerSecond() / integration) / conf.getNrSamplesPerThread(), observation.getNrDMs());
      cl::NDRange local(conf.getNrSamplesPerBlock(), 1);
      kernel->setArg(0, input_d);
      kernel->setArg(1, output_d);
      try {
        // Warm-up run
        clQueues->at(clDeviceID)[0].finish();
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
        event.wait();
        // Tuning runs
        for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
          timer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
          event.wait();
          timer.stop();
        }
      } catch ( cl::Error & err ) {
        std::cerr << "OpenCL error kernel execution (";
        std::cerr << conf.print() << "): ";
        std::cerr << isa::utils::toString(err.err()) << "." << std::endl;
        delete kernel;
        if ( err.err() == -4 || err.err() == -61 ) {
          return -1;
        }
        reInit = true;
        break;
      }
      delete kernel;

      std::cout << observation.getNrDMs() << " " << observation.getNrSamplesPerSecond() << " " << integration << " ";
      std::cout << conf.print() << " ";
      std::cout << std::setprecision(3);
      std::cout << gflops / timer.getAverageTime() << " ";
      std::cout << gbs / timer.getAverageTime() << " ";
      std::cout << std::setprecision(6);
      std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " ";
      std::cout << timer.getCoefficientOfVariation() <<  std::endl;
    }
  }

	std::cout << std::endl;

	return 0;
}

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, cl::Buffer * input_d, const unsigned int input_size, cl::Buffer * output_d, const unsigned int output_size) {
  try {
    *input_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, input_size * sizeof(dataType), 0, 0);
    *output_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, output_size * sizeof(dataType), 0, 0);
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString(err.err()) << "." << std::endl;
    throw;
  }
}

