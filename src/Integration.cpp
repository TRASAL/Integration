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

#include <Integration.hpp>

namespace PulsarSearch {

integrationConf::integrationConf() : KernelConf(), subbandDedispersion(false) {}

integrationConf::~integrationConf() {}

std::string integrationConf::print() const {
  return std::to_string(subbandDedispersion) + " " + isa::OpenCL::KernelConf::print();
}

void readTunedIntegrationConf(tunedIntegrationConf & tunedConf, const std::string & confFilename) {
  unsigned int dim0 = 0;
  unsigned int integration = 0;
  std::string temp;
  std::string deviceName;
  PulsarSearch::integrationConf * parameters = 0;
  std::ifstream confFile;

  confFile.open(confFilename);
  if ( !confFile ) {
    throw FileError("Impossible to open " + confFilename);
  }
  while ( ! confFile.eof() ) {
    unsigned int splitPoint = 0;

    std::getline(confFile, temp);
    if ( ! std::isalpha(temp[0]) ) {
      continue;
    }
    parameters = new PulsarSearch::integrationConf();

    splitPoint = temp.find(" ");
    deviceName = temp.substr(0, splitPoint);
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    dim0 = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    integration = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setSubbandDedispersion(isa::utils::castToType< std::string, bool >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrThreadsD0(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    parameters->setNrThreadsD0(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrThreadsD1(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrThreadsD2(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrItemsD0(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters->setNrItemsD1(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    parameters->setNrItemsD2(isa::utils::castToType< std::string, unsigned int >(temp));

    if ( tunedConf.count(deviceName) == 0 ) {
      std::map< unsigned int, std::map< unsigned int, PulsarSearch::integrationConf * > * >  * externalContainer = new std::map< unsigned int, std::map< unsigned int, PulsarSearch::integrationConf * > * >();
      std::map< unsigned int, PulsarSearch::integrationConf * > * internalContainer = new std::map< unsigned int, PulsarSearch::integrationConf * >();

      internalContainer->insert(std::make_pair(integration, parameters));
      externalContainer->insert(std::make_pair(dim0, internalContainer));
      tunedConf.insert(std::make_pair(deviceName, externalContainer));
    } else if ( tunedConf.at(deviceName)->count(dim0) == 0 ) {
      std::map< unsigned int, PulsarSearch::integrationConf * > * internalContainer = new std::map< unsigned int, PulsarSearch::integrationConf * >();

      internalContainer->insert(std::make_pair(integration, parameters));
      tunedConf.at(deviceName)->insert(std::make_pair(dim0, internalContainer));
    } else {
      tunedConf.at(deviceName)->at(dim0)->insert(std::make_pair(integration, parameters));
    }
  }
  confFile.close();
}

} // PulsarSearch

