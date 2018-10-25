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
  unsigned int padding = 0;
  unsigned int integration = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  uint64_t wrongSamples = 0;
  Integration::integrationConf conf;
  AstroData::Observation observation;

  try {
    isa::utils::ArgumentList args(argc, argv);
    inPlace = args.getSwitch("-in_place");
    if ( !inPlace )
    {
      DMsSamples = args.getSwitch("-dms_samples");
      bool samplesDMs = args.getSwitch("-samples_dms");
      if ( (DMsSamples && samplesDMs) || (!DMsSamples && !samplesDMs) ) {
        std::cerr << "-dms_samples and -samples_dms are mutually exclusive." << std::endl;
        return 1;
      }
    }
    printCode = args.getSwitch("-print_code");
    printResults = args.getSwitch("-print_results");
    random = args.getSwitch("-random");
    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    integration = args.getSwitchArgument< unsigned int >("-integration");
    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threadsD0"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-itemsD0"));
    conf.setSubbandDedispersion(args.getSwitch("-subband"));
    if ( conf.getSubbandDedispersion() ) {
      observation.setDMRange(args.getSwitchArgument< unsigned int >("-subbanding_dms"), 0.0f, 0.0f, true);
    } else {
      observation.setDMRange(1, 0.0f, 0.0f, true);
    }
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0f, 0.0f);
    observation.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
    observation.setNrSynthesizedBeams(args.getSwitchArgument< unsigned int >("-beams"));
    if ( inPlace )
    {
      conf.setIntType(args.getSwitchArgument<unsigned int>("-int_type"));
    }
  } catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << "Usage: " << argv[0] << " [-dms_samples | -samples_dms] [-print_code] [-print_results] [-random] -opencl_platform ... -opencl_device ... -padding ... -integration ... -threadsD0 ... -itemsD0 ... [-in_place] [-subband] -beams ... -samples ... -dms ..." << std::endl;
    std::cerr << " -subband -subbanding_dms ..." << std::endl;
    std::cerr << " -in_place -int_type ..." << std::endl;
    return 1;
  }

  // Initialize OpenCL
  cl::Context * clContext = new cl::Context();
  std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
  std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
  std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

  // Allocate memory
  cl::Buffer input_d;
  cl::Buffer output_d;
  std::vector< dataType > input;
  std::vector< dataType > output;
  std::vector< dataType > output_control;

  if ( DMsSamples )
  {
    input.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType)));
    output.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType)));
    output_control.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType)));
  }
  else if ( inPlace )
  {
    input.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType)));
    output_control.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType)));
  }
  else 
  {
    input.resize(observation.getNrSynthesizedBeams() * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType)));
    output.resize(observation.getNrSynthesizedBeams() * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType)));
    output_control.resize(observation.getNrSynthesizedBeams() * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType)));
  }

  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, input.size() * sizeof(dataType), 0, 0);
    if ( !inPlace )
    {
      output_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, output.size() * sizeof(dataType), 0, 0);
    }
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error allocating memory: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  srand(time(0));
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    if ( printResults ) {
      std::cout << "Beam: " << beam << std::endl;
    }
    if ( DMsSamples || inPlace )
    {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          if ( printResults ) {
            std::cout << "DM: " << (subbandDM * observation.getNrDMs()) + dm << " -- ";
          }
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
            if ( random ) {
              input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + sample] = rand() % 10;
            } else {
              input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + sample] = sample % 10;
            }
            if ( printResults ) {
              std::cout << input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(dataType))) + sample] << " ";
            }
          }
          if ( printResults ) {
            std::cout << std::endl;
          }
        }
      }
    }
    else
    {
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
        if ( printResults ) {
          std::cout << "Sample: " << sample << " -- ";
        }
        for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
            if ( random ) {
              input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(dataType))) + dm] = rand() % 10;
            } else {
              input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(dataType))) + dm] = sample % 10;
            }
            if ( printResults ) {
              std::cout << input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(dataType))) + dm] << " ";
            }
          }
          if ( printResults ) {
            std::cout << std::endl;
          }
        }
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
    std::cerr << "OpenCL error H2D transfer: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  std::string * code;
  if ( DMsSamples )
  {
    code = Integration::getIntegrationDMsSamplesOpenCL< dataType >(conf, observation, dataName, integration, padding);
  }
  else if ( inPlace )
  {
    code = Integration::getIntegrationAfterDedispersionInPlaceOpenCL<dataType>(conf, observation, dataName, integration, padding);
  }
  else
  {
    code = Integration::getIntegrationSamplesDMsOpenCL< dataType >(conf, observation, dataName, integration, padding);
  }
  cl::Kernel * kernel;
  if ( printCode ) {
    std::cout << *code << std::endl;
  }
  try {
    if ( DMsSamples )
    {
      kernel = isa::OpenCL::compile("integrationDMsSamples" + std::to_string(integration), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    }
    else if ( inPlace )
    {
      kernel = isa::OpenCL::compile("integration" + std::to_string(integration), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    }
    else
    {
      kernel = isa::OpenCL::compile("integrationSamplesDMs" + std::to_string(integration), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    }
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global;
    cl::NDRange local;

    if ( DMsSamples || inPlace )
    {
      Integration::integrationDMsSamples(conf.getSubbandDedispersion(), observation, integration, padding, input, output_control);
    }
    else
    {
      Integration::integrationSamplesDMs(conf.getSubbandDedispersion(), observation, integration, padding, input, output_control);
    }
    if ( DMsSamples )
    {
      global = cl::NDRange(conf.getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / integration) / conf.getNrItemsD0()), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
      local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
    }
    else if ( inPlace )
    {
      global = cl::NDRange(conf.getNrThreadsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
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
    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
    if ( inPlace )
    {
      clQueues->at(clDeviceID)[0].enqueueReadBuffer(input_d, CL_TRUE, 0, input.size() * sizeof(dataType), reinterpret_cast< void * >(input.data()));
    }
    else
    {
      clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(dataType), reinterpret_cast< void * >(output.data()));
    }
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error kernel execution: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    if ( printResults ) {
      std::cout << "Beam: " << beam << std::endl;
    }
    if ( DMsSamples )
    {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          if ( printResults ) {
            std::cout << "DM: " << (subbandDM * observation.getNrDMs()) + dm << " -- ";
          }
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / integration; sample++ ) {
            if ( !isa::utils::same(output_control[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + sample], output[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + sample]) ) {
              wrongSamples++;
            }
            if ( printResults ) {
              std::cout << output_control[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + sample] << "," << output[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + sample] << " ";
            }
          }
          if ( printResults )
          {
            std::cout << std::endl;
          }
        }
      }
    }
    else if ( inPlace )
    {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          if ( printResults ) {
            std::cout << "DM: " << (subbandDM * observation.getNrDMs()) + dm << " -- ";
          }
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / integration; sample++ ) {
            if ( !isa::utils::same(output_control[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + sample], input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(dataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(dataType))) + sample]) ) {
              wrongSamples++;
            }
            if ( printResults ) {
              std::cout << output_control[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / integration, padding / sizeof(dataType))) + sample] << "," << input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(dataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch(), padding / sizeof(dataType))) + sample] << " ";
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
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / integration; sample++ ) {
      if ( printResults ) {
        std::cout << "Sample: " << sample << " -- ";
      }
        for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMs(true); subbandDM++ ) {
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
            if ( !isa::utils::same(output_control[(beam * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(dataType))) + dm], output[(beam * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(dataType))) + dm]) ) {
              wrongSamples++;
            }
            if ( printResults ) {
              std::cout << output_control[(beam * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(dataType))) + dm] << "," << output[(beam * (observation.getNrSamplesPerBatch() / integration) * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(dataType))) + (subbandDM * observation.getNrDMs(false, padding / sizeof(dataType))) + dm] << " ";
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

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast< uint64_t >(observation.getNrDMs(true) * observation.getNrDMs()) * (observation.getNrSamplesPerBatch() / integration)) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

  return 0;
}

