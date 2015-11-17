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

integrationSamplesDMsConf::integrationSamplesDMsConf() {}

integrationSampleDMsConf::~integrationSamplesDMsConf() {}

std::string integrationSamplesDMsConf::print() const {
  return isa::utils::toString(nrSamplesPerBlock) + " " + isa::utils::toString(nrSamplesPerThread);
}

void readTunedIntegrationSamplesDMsConf(tunedIntegrationSamplesDMsConf & tunedConf, const std::string & confFilename) {
	std::string temp;
	std::ifstream confFile(confFilename);

	while ( ! confFile.eof() ) {
		unsigned int splitPoint = 0;

		std::getline(confFile, temp);
		if ( ! std::isalpha(temp[0]) ) {
			continue;
		}
		std::string deviceName;
		unsigned int nrSamples = 0;
		unsigned int integration = 0;
    PulsarSearch::integrationSamplesDMsConf parameters;

		splitPoint = temp.find(" ");
		deviceName = temp.substr(0, splitPoint);
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		nrSamples = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		integration = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		parameters.setNrSamplesPerBlock(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
		temp = temp.substr(splitPoint + 1);
		splitPoint = temp.find(" ");
		parameters.setNrSamplesPerThread(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
		if ( tunedConf.count(deviceName) == 0 ) {
      std::map< unsigned int, std::map< unsigned int, PulsarSearch::integrationSamplesDMsConf > > externalContainer;
      std::map< unsigned int, PulsarSearch::integrationSamplesDMsConf > internalContainer;

			internalContainer.insert(std::make_pair(integration, parameters));
			externalContainer.insert(std::make_pair(nrSamples, internalContainer));
			tunedConf.insert(std::make_pair(deviceName, externalContainer));
		} else if ( tunedConf[deviceName].count(nrSamples) == 0 ) {
      std::map< unsigned int, PulsarSearch::integrationSamplesDMsConf > internalContainer;

			internalContainer.insert(std::make_pair(integration, parameters));
			tunedConf[deviceName].insert(std::make_pair(nrSamples, internalContainer));
		} else {
			tunedConf[deviceName][nrSamples].insert(std::make_pair(integration, parameters));
		}
  }
}

} // PulsarSearch

