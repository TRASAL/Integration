// Copyright 2017 Netherlands Institute for Radio Astronomy (ASTRON)
// Copyright 2017 Netherlands eScience Center
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


void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, cl::Buffer * input_d, const unsigned int input_size, cl::Buffer * output_d, const unsigned int output_size);

int main(int argc, char * argv[]) {
  bool reinitializeDeviceMemory = true;
  bool DMsSamples = false;
  bool inPlace = false;
  bool beforeDedispersion = false;
  bool bestMode = false;
  unsigned int padding = 0;
  unsigned int integration = 0;
  unsigned int nrIterations = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  unsigned int minThreads = 0;
  unsigned int maxThreads = 0;
  unsigned int maxItems = 0;
  unsigned int vectorWidth = 0;
  double bestGFLOPs = 0.0;
  AstroData::Observation observation;
  Integration::integrationConf conf;
  Integration::integrationConf bestConf;
  cl::Event event;

  try
  {
    isa::utils::ArgumentList args(argc, argv);
    // Modes
    inPlace = args.getSwitch("-in_place");
    if ( inPlace )
    {
      beforeDedispersion = args.getSwitch("-before_dedispersion");
      bool afterDedispersion = args.getSwitch("-after_dedispersion");
      if ( (beforeDedispersion && afterDedispersion) || (!beforeDedispersion && !afterDedispersion) )
      {
        std::cerr << "-before_dedispersion and -after_dedispersion are mutually exclusive." << std::endl;
        return 1;
      }
    }
    else
    {
      DMsSamples = args.getSwitch("-dms_samples");
      bool samplesDMs = args.getSwitch("-samples_dms");
      if ( (DMsSamples && samplesDMs) || (!DMsSamples && !samplesDMs) )
      {
        std::cerr << "-dms_samples and -samples_dms are mutually exclusive." << std::endl;
        return 1;
      }
    }
    // OpenCL
    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    // Tuning
    bestMode = args.getSwitch("-best");
    nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
    minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    // Scenario
    padding = args.getSwitchArgument< unsigned int >("-padding");
    integration = args.getSwitchArgument< unsigned int >("-integration");
    vectorWidth = args.getSwitchArgument< unsigned int >("-vector");
    observation.setNrSynthesizedBeams(args.getSwitchArgument< unsigned int >("-beams"));
    observation.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
    if ( inPlace && beforeDedispersion )
    {
      observation.setFrequencyRange(1, args.getSwitchArgument<unsigned int>("-channels"), 0.0f, 0.0f);
      observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerBatch());
      observation.setNrBeams(observation.getNrSynthesizedBeams());
    }
    else
    {
      conf.setSubbandDedispersion(args.getSwitch("-subband"));
      if ( conf.getSubbandDedispersion() )
      {
        observation.setDMRange(args.getSwitchArgument< unsigned int >("-subbanding_dms"), 0.0f, 0.0f, true);
      }
      else
      {
        observation.setDMRange(1, 0.0f, 0.0f, true);
      }
      observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0f, 0.0f);
    }
  }
  catch ( isa::utils::EmptyCommandLine & err )
  {
    std::cerr << argv[0] << " [-in_place] [-dms_samples | -samples_dms] [-best] -iterations ... -opencl_platform ... -opencl_device ... -padding ... -integration ... -min_threads ... -max_threads ... -max_items ... -vector ... [-subband] -beams ... -samples ... -dms ... " << std::endl;
    std::cerr << "\t -subband : -subbanding_dms ..." << std::endl;
    std::cerr << " -in_place [-before_dedispersion | -after_dedispersion]" << std::endl;
    std::cerr << " -before_dedispersion -channels ..." << std::endl;
    return 1;
  }
  catch ( std::exception & err )
  {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Allocate host memory
  std::vector<dataType> input, output;
  if ( inPlace )
  {
    if ( beforeDedispersion )
    {
      input.resize(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(dataType)));
    }
    else
    {
      input.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType)));
    }
  }
  else
  {
    if ( DMsSamples )
    {
      input.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType)));
      output.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType)));
    }
    else
    {
      input.resize(observation.getNrSynthesizedBeams() * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType)));
      output.resize(observation.getNrSynthesizedBeams() * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType)));
    }
  }

  cl::Context clContext;
  std::vector<cl::Platform> * clPlatforms = new std::vector<cl::Platform>();
  std::vector<cl::Device> * clDevices = new std::vector<cl::Device>();
  std::vector<std::vector<cl::CommandQueue>> * clQueues = new std::vector<std::vector<cl::CommandQueue>>();
  cl::Buffer input_d;
  cl::Buffer output_d;

  if ( !bestMode )
  {
    std::cout << std::fixed << std::endl;
    if ( inPlace && beforeDedispersion )
    {
      std::cout << "# nrBeams nrChannels nrSamples integration *configuration* GFLOP/s GB/s time stdDeviation COV" << std::endl << std::endl;
    }
    else
    {
      std::cout << "# nrBeams nrDMs nrSamples integration *configuration* GFLOP/s GB/s time stdDeviation COV" << std::endl << std::endl;
    }
  }

  for ( unsigned int threads = minThreads; threads <= maxThreads; )
  {
    conf.setNrThreadsD0(threads);
    if ( DMsSamples || inPlace )
    {
      threads *= 2;
    }
    else
    {
      threads++;
    }
    if ( conf.getNrThreadsD0() % vectorWidth != 0 )
    {
      continue;
    }
    for ( unsigned int itemsPerThread = 1; itemsPerThread <= maxItems; itemsPerThread++ )
    {
      conf.setNrItemsD0(itemsPerThread);
      if ( inPlace && beforeDedispersion )
      {
        if ( conf.getNrItemsD0() + 2 >= maxItems )
        {
          break;
        }
        else if ( (conf.getNrThreadsD0() * conf.getNrItemsD0() * integration) > observation.getNrSamplesPerDispersedBatch() )
        {
          break;
        }
        else if ( (observation.getNrSamplesPerDispersedBatch() % (integration * conf.getNrItemsD0())) != 0 )
        {
          continue;
        }
      }
      else if ( inPlace && !beforeDedispersion)
      {
        if ( conf.getNrItemsD0() + 2 >= maxItems )
        {
          break;
        }
        else if ( (conf.getNrThreadsD0() * conf.getNrItemsD0() * integration) > observation.getNrSamplesPerBatch() )
        {
          break;
        }
        else if ( (observation.getNrSamplesPerBatch() % (integration * conf.getNrItemsD0())) != 0 )
        {
          continue;
        }
      }
      else if ( DMsSamples )
      {
        if ( (observation.getNrSamplesPerBatch() % (integration * conf.getNrItemsD0())) != 0 )
        {
          continue;
        }
      }
      else
      {
        if ( observation.getNrDMs() % (conf.getNrThreadsD0() * conf.getNrItemsD0()) != 0 )
        {
          continue;
        }
      }
      for ( unsigned int intType = 0; intType < 2; intType++ )
      {
        conf.setIntType(intType);
        // Generate kernel
        double gflops, gbs;
        if ( inPlace && beforeDedispersion )
        {
          gflops = isa::utils::giga(observation.getNrBeams() * static_cast<uint64_t>(observation.getNrChannels()) * observation.getNrSamplesPerDispersedBatch());
          gbs = isa::utils::giga((observation.getNrBeams() * static_cast<uint64_t>(observation.getNrChannels()) * observation.getNrSamplesPerDispersedBatch()) + (observation.getNrBeams() * static_cast<uint64_t>(observation.getNrChannels()) * (observation.getNrSamplesPerDispersedBatch() / integration)));
        }
        else
        {
          gflops = isa::utils::giga(observation.getNrSynthesizedBeams() * static_cast<uint64_t>(observation.getNrDMs(true) * observation.getNrDMs()) * observation.getNrSamplesPerBatch());
          gbs = isa::utils::giga((observation.getNrSynthesizedBeams() * static_cast<uint64_t>(observation.getNrDMs(true) * observation.getNrDMs()) * observation.getNrSamplesPerBatch()) + (observation.getNrSynthesizedBeams() * static_cast<uint64_t>(observation.getNrDMs(true) * observation.getNrDMs()) * (observation.getNrSamplesPerBatch() / integration)));
        }
        isa::utils::Timer timer;
        cl::Kernel * kernel;

        std::string * code;
        if ( inPlace && beforeDedispersion )
        {
          code = Integration::getIntegrationBeforeDedispersionInPlaceOpenCL<dataType>(conf, observation, dataName, integration, padding);
        }
        else if ( inPlace && !beforeDedispersion )
        {
          code = Integration::getIntegrationAfterDedispersionInPlaceOpenCL<dataType>(conf, observation, dataName, integration, padding);
        }
        else if ( DMsSamples )
        {
          code = Integration::getIntegrationDMsSamplesOpenCL< dataType >(conf, observation, dataName, integration, padding);
        }
        else
        {
          code = Integration::getIntegrationSamplesDMsOpenCL< dataType >(conf, observation, dataName, integration, padding);
        }
        if ( reinitializeDeviceMemory )
        {
          delete clQueues;
          clQueues = new std::vector< std::vector < cl::CommandQueue > >();
          isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
          try
          {
            initializeDeviceMemory(clContext, &(clQueues->at(clDeviceID)[0]), &input_d, input.size(), &output_d, output.size());
          }
          catch ( cl::Error & err )
          {
            std::cerr << "Error in memory allocation: ";
            std::cerr << std::to_string(err.err()) << "." << std::endl;
            return -1;
          }
          reinitializeDeviceMemory = false;
        }
        try
        {
          if ( inPlace )
          {
            kernel = isa::OpenCL::compile("integration" + std::to_string(integration), *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
          }
          else if ( DMsSamples )
          {
            kernel = isa::OpenCL::compile("integrationDMsSamples" + std::to_string(integration), *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
          }
          else
          {
            kernel = isa::OpenCL::compile("integrationSamplesDMs" + std::to_string(integration), *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
          }
        }
        catch ( isa::OpenCL::OpenCLError & err )
        {
          std::cerr << err.what() << std::endl;
          delete code;
          break;
        }
        delete code;

        cl::NDRange global;
        cl::NDRange local;
        if ( inPlace && beforeDedispersion )
        {
          global = cl::NDRange(conf.getNrThreadsD0(), observation.getNrChannels(), observation.getNrBeams());
          local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
        }
        else if ( inPlace && !beforeDedispersion )
        {
          global = cl::NDRange(conf.getNrThreadsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
          local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
        }
        else if ( DMsSamples )
        {
          global = cl::NDRange(conf.getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / integration) / conf.getNrItemsD0()), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
          local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
        }
        else
        {
          global = cl::NDRange((observation.getNrDMs(true) * observation.getNrDMs())/ conf.getNrItemsD0(), observation.getNrSamplesPerBatch() / integration, observation.getNrSynthesizedBeams());
          local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
        }
        kernel->setArg(0, input_d);
        if ( !inPlace )
        {
          kernel->setArg(1, output_d);
        }
        try
        {
          // Warm-up run
          clQueues->at(clDeviceID)[0].finish();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
          event.wait();
          // Tuning runs
          for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ )
          {
            timer.start();
            clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
            event.wait();
            timer.stop();
          }
        }
        catch ( cl::Error & err )
        {
          std::cerr << "OpenCL error kernel execution (";
          std::cerr << conf.print() << "): ";
          std::cerr << std::to_string(err.err()) << "." << std::endl;
          delete kernel;
          if ( err.err() == -4 || err.err() == -61 )
          {
            return -1;
          }
          reinitializeDeviceMemory = true;
          break;
        }
        delete kernel;

        if ( (gflops / timer.getAverageTime()) > bestGFLOPs )
        {
          bestGFLOPs = gflops / timer.getAverageTime();
          bestConf = conf;
        }
        if ( !bestMode )
        {
          if ( inPlace && beforeDedispersion )
          {
            std::cout << observation.getNrBeams() << " " << observation.getNrChannels() << " " << observation.getNrSamplesPerDispersedBatch() << " " << integration << " ";
          }
          else
          {
            std::cout << observation.getNrSynthesizedBeams() << " " << observation.getNrDMs(true) * observation.getNrDMs() << " " << observation.getNrSamplesPerBatch() << " " << integration << " ";
          }
          std::cout << conf.print() << " ";
          std::cout << std::setprecision(3);
          std::cout << gflops / timer.getAverageTime() << " ";
          std::cout << gbs / timer.getAverageTime() << " ";
          std::cout << std::setprecision(6);
          std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " ";
          std::cout << timer.getCoefficientOfVariation() <<  std::endl;
        }
      }
    }
  }

  if ( bestMode )
  {
    std::cout << observation.getNrDMs(true) * observation.getNrDMs() << " " << integration << " " << bestConf.print() << std::endl;
  }
  else
  {
    std::cout << std::endl;
  }

  return 0;
}

void initializeDeviceMemory(cl::Context & clContext, cl::CommandQueue * clQueue, cl::Buffer * input_d, const unsigned int input_size, cl::Buffer * output_d, const unsigned int output_size) {
  try
  {
    *input_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, input_size * sizeof(dataType), 0, 0);
    if ( output_size > 0 )
    {
      *output_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, output_size * sizeof(dataType), 0, 0);
    }
    clQueue->finish();
  }
  catch ( cl::Error & err )
  {
    std::cerr << "OpenCL error: " << std::to_string(err.err()) << "." << std::endl;
    throw;
  }
}

