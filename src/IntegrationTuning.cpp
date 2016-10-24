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
  bool DMsSamples = false;
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
  PulsarSearch::integrationConf conf;
  cl::Event event;

  try {
    isa::utils::ArgumentList args(argc, argv);
    DMsSamples = args.getSwitch("-dms_samples");
    bool samplesDMs = args.getSwitch("-samples_dms");
    if ( (DMsSamples && samplesDMs) || !(DMSamples && samplesDMs) ) {
      std::cerr << "-dms_samples and -samples_dms are mutually exclusive." << std::endl;
      return 1;
    }
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    integration = args.getSwitchArgument< unsigned int >("-integration");
    minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    vectorWidth = args.getSwitchArgument< unsigned int >("-vector");
    observation.setNrSyntheticBeams(args.getSwitchArgument< unsigned int >("-beams"));
    observation.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
    conf.setSubbandDedispersion(args.getSwitch("-subband"));
    if ( conf.getSubbandDedispersion() ) {
      observation.setDMSubbandingRange(args.getSwitchArgument< unsigned int >("-subbanding_dms"), 0.0f, 0.0f);
    } else {
      observation.setDMSubbandingRange(1, 0.0f, 0.0f);
    }
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0f, 0.0f);
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << argv[0] << " [-dms_samples | -samples_dms] -iterations ... -opencl_platform ... -opencl_device ... -padding ... -integration ... -min_threads ... -max_threads ... -max_items ... -vector ... [-subband] -beams ... -samples ... -dms ... " << std::endl;
    std::cerr << "\t -subband : -subbanding_dms ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Allocate host memory
  std::vector< dataType > input, output;
  if ( DMsSamples ) {
    input = std::vector< dataType >(observation.getNrSyntheticBeams() * observation.getNrDMsSubbanding() * observation.getNrDMs() * observation.getNrSamplesPerPaddedBatch(padding / sizeof(dataType)));
    output = std::vector< dataType >(observation.getNrSyntheticBeams() * observation.getNrDMsSubbanding() * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType)));
  } else {
    input = std::vector< dataType >(observation.getNrSyntheticBeams() * observation.getNrSamplesPerBatch() * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(dataType)));
    output = std::vector< dataType >(observation.getNrSyntheticBeams() * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(dataType)));
  }

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
  std::cout << "# nrBeams nrDMs nrSamples integration *configuration* GFLOP/s GB/s time stdDeviation COV" << std::endl << std::endl;

  for ( unsigned int threads = minThreads; threads <= maxThreads; threads++) {
    conf.setNrThreadsD0(threads);
    if ( conf.getNrThreadsD0() % vectorWidth != 0 ) {
      continue;
    }

    for ( unsigned int itemsPerThread = 1; itemsPerThread <= maxItems; itemsPerThread++ ) {
      conf.setNrItemsD0(itemsPerThread);
      if ( DMsSamples ) {
        if ( (observation.getNrSamplesPerBatch() % (integration * conf.getNrItemsD0())) != 0 ) {
          continue;
        }
      } else {
        if ( observation.getNrDMs() % (conf.getNrThreadsD0() * conf.getNrItemsD0()) != 0 ) {
          continue;
        }
      }

      // Generate kernel
      double gflops = isa::utils::giga(observation.getNrSyntheticBeams() * static_cast< uint64_t >(observation.getNrDMsSubbanding() * observation.getNrDMs()) * observation.getNrSamplesPerBatch());
      double gbs = isa::utils::giga((observation.getNrSyntheticBeams() * static_cast< uint64_t >(observation.getNrDMsSubbanding() * observation.getNrDMs()) * observation.getNrSamplesPerBatch()) + (observation.getNrSyntheticBeams() * static_cast< uint64_t >(observation.getNrDMsSubbanding() * observation.getNrDMs()) * (observation.getNrSamplesPerBatch() / integration)));
      isa::utils::Timer timer;
      cl::Kernel * kernel;

      std::string * code;
      if ( DMsSamples ) {
        code = PulsarSearch::getIntegrationDMsSamplesOpenCL< dataType >(conf, observation, dataName, integration, padding);
      } else {
        code = PulsarSearch::getIntegrationSamplesDMsOpenCL< dataType >(conf, observation, dataName, integration, padding);
      }
      if ( reInit ) {
        delete clQueues;
        clQueues = new std::vector< std::vector < cl::CommandQueue > >();
        isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
        try {
          initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &input_d, input.size(), &output_d, output.size());
        } catch ( cl::Error & err ) {
          std::cerr << "Error in memory allocation: ";
          std::cerr << std::to_string(err.err()) << "." << std::endl;
          return -1;
        }
        reInit = false;
      }
      try {
        if ( DMsSamples ) {
          kernel = isa::OpenCL::compile("integrationDMsSamples" + std::to_string(integration), *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
        } else {
          kernel = isa::OpenCL::compile("integrationSamplesDMs" + std::to_string(integration), *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
        }
      } catch ( isa::OpenCL::OpenCLError & err ) {
        std::cerr << err.what() << std::endl;
        delete code;
        break;
      }
      delete code;

      cl::NDRange global;
      cl::NDRange local;
      if ( DMsSamples ) {
        global = cl::NDRange(conf.getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / integration) / conf.getNrItemsD0()), observation.getNrDMsSubbanding() * observation.getNrDMs(), observation.getNrSyntheticBeams());
        local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
      } else if ( samplesDMs ) {
        global = cl::NDRange((observation.getNrDMsSubbanding() * observation.getNrDMs())/ conf.getNrItemsD0(), observation.getNrSamplesPerBatch() / integration, observation.getNrSyntheticBeams());
        local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
      }
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
        std::cerr << std::to_string(err.err()) << "." << std::endl;
        delete kernel;
        if ( err.err() == -4 || err.err() == -61 ) {
          return -1;
        }
        reInit = true;
        break;
      }
      delete kernel;

      std::cout << observation.getNrSyntheticBeams() << " " << observation.getNrDMsSubbanding() * observation.getNrDMs() << " " << observation.getNrSamplesPerBatch() << " " << integration << " ";
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
    std::cerr << "OpenCL error: " << std::to_string(err.err()) << "." << std::endl;
    throw;
  }
}

