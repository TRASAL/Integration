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
#include <exception>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <Integration.hpp>

int main(int argc, char * argv[]) {
  bool DMsSamples = false;
  bool samplesDMs = false;
  unsigned int padding = 0;
  unsigned int integration = 0;
  PulsarSearch::integrationConf conf;
  AstroData::Observation observation;

  try {
    isa::utils::ArgumentList args(argc, argv);
    DMsSamples = args.getSwitch("-dms_samples");
    samplesDMs = args.getSwitch("-samples_dms");
    if ( samplesDMs ) {
      observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0f, 0.0f);
    }
    padding = args.getSwitchArgument< unsigned int >("-padding");
    integration = args.getSwitchArgument< unsigned int >("-integration");
    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-samples_per_block"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-samples_per_thread"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
  } catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << "Usage: " << argv[0] << " [-dms_samples] [-samples_dms] -padding ... -integration ... -samples_per_block ... -samples_per_thread ... -samples ..." << std::endl;
    std::cerr << "\t -samples_dms -dms ..." << std::endl;
		return 1;
	}

	// Generate kernel
  if ( DMsSamples ) {
    std::string * code = PulsarSearch::getIntegrationDMsSamplesOpenCL< dataType >(conf, observation.getNrSamplesPerSecond(), dataName, integration, padding);
    std::cout << *code << std::endl;
  }
  if ( samplesDMs ) {
    std::string * code = PulsarSearch::getIntegrationSamplesDMsOpenCL< dataType >(conf, observation, dataName, integration, padding);
    std::cout << *code << std::endl;
  }

  return 0;
}

