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
#include <ctime>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Integration.hpp>


int main(int argc, char *argv[]) {
  bool printCode = false;
  bool printResults = false;
  bool random = false;
  bool DMsSamples = false;
  bool inPlace = false;
  bool beforeDedispersion = false;
  unsigned int padding = 0;
  unsigned int integration = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  uint64_t wrongSamples = 0;
  Integration::integrationConf conf;
  AstroData::Observation observation;

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
    printCode = args.getSwitch("-print_code");
    printResults = args.getSwitch("-print_results");
    random = args.getSwitch("-random");
    // OpenCL
    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    // Configuration
    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threadsD0"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-itemsD0"));
    conf.setIntType(args.getSwitchArgument<unsigned int>("-int_type"));
    // Scenario
    padding = args.getSwitchArgument< unsigned int >("-padding");
    integration = args.getSwitchArgument< unsigned int >("-integration");
    observation.setNrSynthesizedBeams(args.getSwitchArgument< unsigned int >("-beams"));
    observation.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
    if ( inPlace && beforeDedispersion )
    {
      observation.setFrequencyRange(1, args.getSwitchArgument<unsigned int>("-channels"), 0.0f, 0.0f);
      observation.setNrBeams(observation.getNrSynthesizedBeams());
      conf.setSubbandDedispersion(args.getSwitch("-subband"));
      if ( conf.getSubbandDedispersion() )
      {
        observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerBatch(), true);
      }
      else
      {
        observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerBatch());
      }
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
  catch  ( isa::utils::SwitchNotFound & err )
  {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  catch ( std::exception & err )
  {
    std::cerr << "Usage: " << argv[0] << " [-in_place] [-dms_samples | -samples_dms] [-print_code] [-print_results] [-random] -opencl_platform ... -opencl_device ... -padding ... -int_type ... -integration ... -threadsD0 ... -itemsD0 ... [-subband] -beams ... -samples ... -dms ..." << std::endl;
    std::cerr << " -subband -subbanding_dms ..." << std::endl;
    std::cerr << " -in_place [-before_dedispersion | -after_dedispersion]" << std::endl;
    std::cerr << " -before_dedispersion -channels ..." << std::endl;
    return 1;
  }

  // Initialize OpenCL
  isa::OpenCL::OpenCLRunTime openCLRunTime;

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, openCLRunTime);

  // Allocate memory
  cl::Buffer input_d;
  cl::Buffer output_d;
  std::vector<BeforeDedispersionNumericType> input_before;
  std::vector<AfterDedispersionNumericType> input_after;
  std::vector<AfterDedispersionNumericType> output;
  std::vector<BeforeDedispersionNumericType> output_control_before;
  std::vector<AfterDedispersionNumericType> output_control_after;

  if ( inPlace )
  {
    if ( beforeDedispersion )
    {
      if ( conf.getSubbandDedispersion() )
      {
        input_before.resize(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, padding / sizeof(BeforeDedispersionNumericType)));
        output_control_before.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true), padding / sizeof(BeforeDedispersionNumericType)));
      }
      else
      {
        input_before.resize(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(BeforeDedispersionNumericType)));
        output_control_before.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(), padding / sizeof(BeforeDedispersionNumericType)));
      }
    }
    else
    {
      input_after.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType)));
      output_control_after.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType)));
    }
  }
  else
  {
    if ( DMsSamples )
    {
      input_after.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType)));
      output.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType)));
      output_control_after.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType)));
    }
    else 
    {
      input_after.resize(observation.getNrSynthesizedBeams() * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType)));
      output.resize(observation.getNrSynthesizedBeams() * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType)));
      output_control_after.resize(observation.getNrSynthesizedBeams() * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType)));
    }
  }

  try {
    if ( inPlace && beforeDedispersion )
    {
      input_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_WRITE, input_before.size() * sizeof(BeforeDedispersionNumericType), 0, 0);
    }
    else
    {
      input_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_WRITE, input_after.size() * sizeof(AfterDedispersionNumericType), 0, 0);
    }
    if ( !inPlace )
    {
      output_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_WRITE, output.size() * sizeof(AfterDedispersionNumericType), 0, 0);
    }
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error allocating memory: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  // Generation of test data
  srand(time(0));
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ )
  {
    if ( printResults )
    {
      std::cout << "Beam: " << beam << std::endl;
    }
    if ( inPlace && beforeDedispersion )
    {
      for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ )
      {
        if ( printResults )
        {
          std::cout << "Channel: " << channel << " -- ";
        }
        for ( unsigned int sample = 0; sample < observation.getNrSamplesPerDispersedBatch(); sample++ )
        {
          if ( random )
          {
            input_before[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(BeforeDedispersionNumericType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(BeforeDedispersionNumericType))) + sample] = rand() % 10;
          }
          else
          {
            input_before[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(BeforeDedispersionNumericType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(BeforeDedispersionNumericType))) + sample] = sample % 10;
          }
          if ( printResults )
          {
            std::cout << static_cast<double>(input_before[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(BeforeDedispersionNumericType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(BeforeDedispersionNumericType))) + sample]) << " ";
          }
        }
        if ( printResults )
        {
          std::cout << std::endl;
        }
      }
    }
    else if ( DMsSamples || inPlace )
    {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ )
      {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ )
        {
          if ( printResults )
          {
            std::cout << "DM: " << (subbandDM * observation.getNrDMs()) + dm << " -- ";
          }
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ )
          {
            if ( random )
            {
              input_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + sample] = rand() % 10;
            }
            else
            {
              input_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + sample] = sample % 10;
            }
            if ( printResults )
            {
              std::cout << static_cast<double>(input_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(AfterDedispersionNumericType))) + sample]) << " ";
            }
          }
          if ( printResults )
          {
            std::cout << std::endl;
          }
        }
      }
    }
    else
    {
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ )
      {
        if ( printResults )
        {
          std::cout << "Sample: " << sample << " -- ";
        }
        for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ )
        {
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ )
          {
            if ( random )
            {
              input_after[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + dm] = rand() % 10;
            }
            else
            {
              input_after[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + dm] = sample % 10;
            }
            if ( printResults )
            {
              std::cout << static_cast<double>(input_after[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + dm]) << " ";
            }
          }
          if ( printResults )
          {
            std::cout << std::endl;
          }
        }
      }
    }
    if ( printResults )
    {
      std::cout << std::endl;
    }
  }
  if ( printResults )
  {
    std::cout << std::endl;
  }

  // Copy data structures to device
  try
  {
    if ( beforeDedispersion )
    {
      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input_before.size() * sizeof(beforeDedispersion), reinterpret_cast< void * >(input_before.data()), 0, 0);
    }
    else
    {
      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input_after.size() * sizeof(AfterDedispersionNumericType), reinterpret_cast< void * >(input_after.data()), 0, 0);
    }
  }
  catch ( cl::Error & err )
  {
    std::cerr << "OpenCL error H2D transfer: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  std::string * code;
  if ( inPlace && beforeDedispersion )
  {
    code = Integration::getIntegrationBeforeDedispersionInPlaceOpenCL<BeforeDedispersionNumericType>(conf, observation, BeforeDedispersionDataName, integration, padding);
  }
  else if ( inPlace && !beforeDedispersion )
  {
    code = Integration::getIntegrationAfterDedispersionInPlaceOpenCL<AfterDedispersionNumericType>(conf, observation, AfterDedispersionDataName, integration, padding);
  }
  else if ( DMsSamples )
  {
    code = Integration::getIntegrationDMsSamplesOpenCL<AfterDedispersionNumericType>(conf, observation, AfterDedispersionDataName, integration, padding);
  }
  else
  {
    code = Integration::getIntegrationSamplesDMsOpenCL<AfterDedispersionNumericType>(conf, observation, AfterDedispersionDataName, integration, padding);
  }
  cl::Kernel * kernel;
  if ( printCode ) {
    std::cout << *code << std::endl;
  }
  try
  {
    if ( inPlace )
    {
      kernel = isa::OpenCL::compile("integration" + std::to_string(integration), *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
    }
    else if ( DMsSamples )
    {
      kernel = isa::OpenCL::compile("integrationDMsSamples" + std::to_string(integration), *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
    }
    else
    {
      kernel = isa::OpenCL::compile("integrationSamplesDMs" + std::to_string(integration), *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
    }
  }
  catch ( isa::OpenCL::OpenCLError & err )
  {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global;
    cl::NDRange local;

    if ( inPlace && beforeDedispersion )
    {
      Integration::integrationBeforeDedispersion(observation, integration, padding, input_before, output_control_before);
    }
    else if ( (inPlace && !beforeDedispersion) || DMsSamples )
    {
      Integration::integrationDMsSamples(conf.getSubbandDedispersion(), observation, integration, padding, input_after, output_control_after);
    }
    else
    {
      Integration::integrationSamplesDMs(conf.getSubbandDedispersion(), observation, integration, padding, input_after, output_control_after);
    }
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
      global = cl::NDRange((observation.getNrDMs(true) * observation.getNrDMs()) / conf.getNrItemsD0(), observation.getNrSamplesPerBatch() / integration, observation.getNrSynthesizedBeams());
      local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
    }

    kernel->setArg(0, input_d);
    if ( !inPlace )
    {
      kernel->setArg(1, output_d);
    }
    openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
    if ( inPlace && beforeDedispersion )
    {
      openCLRunTime.queues->at(clDeviceID)[0].enqueueReadBuffer(input_d, CL_TRUE, 0, input_before.size() * sizeof(BeforeDedispersionNumericType), reinterpret_cast< void * >(input_before.data()));
    }
    else if ( inPlace && !beforeDedispersion )
    {
      openCLRunTime.queues->at(clDeviceID)[0].enqueueReadBuffer(input_d, CL_TRUE, 0, input_after.size() * sizeof(AfterDedispersionNumericType), reinterpret_cast< void * >(input_after.data()));
    }
    else
    {
      openCLRunTime.queues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(AfterDedispersionNumericType), reinterpret_cast< void * >(output.data()));
    }
  }
  catch ( cl::Error & err )
  {
    std::cerr << "OpenCL error kernel execution: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  // Checking the output
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ )
  {
    if ( printResults )
    {
      std::cout << "Beam: " << beam << std::endl;
    }
    if ( inPlace && beforeDedispersion )
    {
      for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ )
      {
        if ( printResults )
        {
          std::cout << "Channel: " << channel << " -- ";
        }
        for ( unsigned int sample = 0; sample < observation.getNrSamplesPerDispersedBatch() / integration; sample++ )
        {
          if ( !isa::utils::same(output_control_before[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / integration, padding / sizeof(BeforeDedispersionNumericType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / integration, padding / sizeof(BeforeDedispersionNumericType))) + sample], input_before[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(), padding / sizeof(BeforeDedispersionNumericType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(), padding / sizeof(BeforeDedispersionNumericType))) + sample]) )
          {
            wrongSamples++;
          }
          if ( printResults )
          {
            std::cout << static_cast<double>(output_control_before[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / integration, padding / sizeof(BeforeDedispersionNumericType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / integration, padding / sizeof(BeforeDedispersionNumericType))) + sample]) << "," << static_cast<double>(input_before[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(), padding / sizeof(BeforeDedispersionNumericType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(), padding / sizeof(BeforeDedispersionNumericType))) + sample]) << " ";
          }
        }
        if ( printResults )
        {
          std::cout << std::endl;
        }
      }
    }
    else if ( inPlace && !beforeDedispersion )
    {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ )
      {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ )
        {
          if ( printResults )
          {
            std::cout << "DM: " << (subbandDM * observation.getNrDMs()) + dm << " -- ";
          }
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / integration; sample++ )
          {
            if ( !isa::utils::same(output_control_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + sample], input_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(AfterDedispersionNumericType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(AfterDedispersionNumericType))) + sample]) )
            {
              wrongSamples++;
            }
            if ( printResults )
            {
              std::cout << static_cast<double>(output_control_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + sample]) << "," << static_cast<double>(input_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(AfterDedispersionNumericType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(AfterDedispersionNumericType))) + sample]) << " ";
            }
          }
          if ( printResults )
          {
            std::cout << std::endl;
          }
        }
      }
    }
    else if ( DMsSamples )
    {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ )
      {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ )
        {
          if ( printResults )
          {
            std::cout << "DM: " << (subbandDM * observation.getNrDMs()) + dm << " -- ";
          }
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / integration; sample++ )
          {
            if ( !isa::utils::same(output_control_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + sample], output[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + sample]) )
            {
              wrongSamples++;
            }
            if ( printResults )
            {
              std::cout << static_cast<double>(output_control_after[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + sample]) << "," << static_cast<double>(output[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(AfterDedispersionNumericType))) + sample]) << " ";
            }
          }
          if ( printResults )
          {
            std::cout << std::endl;
          }
        }
      }
    }
    else
    {
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / integration; sample++ )
      {
      if ( printResults )
      {
        std::cout << "Sample: " << sample << " -- ";
      }
        for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ )
        {
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ )
          {
            if ( !isa::utils::same(output_control_after[(beam * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + dm], output[(beam * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + dm]) )
            {
              wrongSamples++;
            }
            if ( printResults )
            {
              std::cout << static_cast<double>(output_control_after[(beam * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + dm]) << "," << static_cast<double>(output[(beam * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(AfterDedispersionNumericType))) + dm]) << " ";
            }
          }
        }
        if ( printResults )
          {
            std::cout << std::endl;
          }
      }
    }
  }

  // Output
  if ( wrongSamples > 0 )
  {
    if ( inPlace && beforeDedispersion )
    {
      std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast<uint64_t>(observation.getNrChannels()) * (observation.getNrSamplesPerDispersedBatch() / integration)) << "%)." << std::endl;
    }
    else
    {
      std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast<uint64_t>(observation.getNrDMs(true) * observation.getNrDMs()) * (observation.getNrSamplesPerBatch() / integration)) << "%)." << std::endl;
    }
  }
  else
  {
    std::cout << "TEST PASSED." << std::endl;
  }

  return 0;
}
