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
  unsigned int padding = 0;
  unsigned int integration = 0;
  PulsarSearch::integrationDMsSamplesConf conf;
  AstroData::Observation observation;

  try {
    isa::utils::ArgumentList args(argc, argv);
    padding = args.getSwitchArgument< unsigned int >("-padding");
    integration = args.getSwitchArgument< unsigned int >("-integration");
    conf.setNrSamplesPerBlock(args.getSwitchArgument< unsigned int >("-sb"));
		conf.setNrSamplesPerThread(args.getSwitchArgument< unsigned int >("-st"));
		observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
	} catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception & err ) {
    std::cerr << "Usage: " << argv[0] << " -padding ... -integration ... -sb ... -st ... -samples ..." << std::endl;
		return 1;
	}

	// Generate kernel
  std::string * code = PulsarSearch::getIntegrationDMsSamplesOpenCL< dataType >(conf, observation.getNrSamplesPerSecond(), dataName, integration, padding);
  std::cout << *code << std::endl;

  return 0;
}

