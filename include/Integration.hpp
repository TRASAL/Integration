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

#include <string>
#include <map>
#include <vector>
#include <fstream>

#include <Observation.hpp>
#include <utils.hpp>

#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

namespace PulsarSearch {

class integrationSamplesDMsConf {
public:
  integrationSamplesDMsConf();
  ~integrationSamplesDMsConf();
  // Get
  unsigned int getNrSamplesPerBlock() const;
  unsigned int getNrSamplesPerThread() const;
  // Set
  void setNrSamplesPerBlock(unsigned int samples);
  void setNrSamplesPerThread(unsigned int samples);
  // utils
  std::string print() const;
private:
  unsigned int nrSamplesPerBlock;
  unsigned int nrSamplesPerThread;
};

typedef std::map< std::string, std::map < unsigned int, std::map< unsigned int, PulsarSearch::integrationSamplesDMsConf > > > tunedIntegrationSamplesDMsConf;

// Sequential
template< typename T > void integrationSamplesDMs(const AstroData::Observation & observation, const unsigned int integration, const unsigned int padding, const std::vector< T > & input, std::vector< T > & output);
// OpenCL
template< typename T > std::string * getIntegrationSamplesDMsOpenCL(const integrationSamplesDMsConf & conf, const AstroData::Observation & observation, const std::string & inputDataName, const unsigned int integration, const unsigned int padding);
// Read configuration files
void readTunedIntegrationSamplesDMsConf(tunedIntegrationSamplesDMsConf & tunedConf, const std::string & confFilename);


// Implementations
inline unsigned int integrationSamplesDMsConf::getNrSamplesPerBlock() const {
  return nrSamplesPerBlock;
}

inline unsigned int integrationSamplesDMsConf::getNrSamplesPerThread() const {
  return nrSamplesPerThread;
}

inline void integrationSamplesDMsConf::setNrSamplesPerBlock(unsigned int samples) {
  nrSamplesPerBlock = samples;
}

inline void integrationSamplesDMsConf::setNrSamplesPerThread(unsigned int samples) {
  nrSamplesPerThread = samples;
}

template< typename T > void integrationSamplesDMs(const AstroData::Observation & observation, const unsigned int integration, const unsigned int padding, const std::vector< T > & input, std::vector< T > & output) {
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample += integration ) {
      T integratedSample = 0;

      for ( unsigned int i = 0; i < integration; i++ ) {
        integratedSample += input[(dm * observation.getNrSamplesPerPaddedSecond(padding / sizeof(T))) + (sample + i)];
      }
      output[(dm * isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(T))) + (sample / integration)] = integratedSample / integration;
    }
  }
}

template< typename T > std::string * getIntegrationSamplesDMsOpenCL(const integrationSamplesDMsConf & conf, const AstroData::Observation & observation, const std::string & dataName, const unsigned int integration, const unsigned int padding) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void integrateSamplesDMs" + isa::utils::toString(integration) + "(__global const " + dataName + " * const restrict input, __global " + dataName + " * const restrict output) {\n"
    "unsigned int dm = get_group_id(1);\n"
    "__local " + dataName + " buffer[" + isa::utils::toString(integration * conf.getNrSamplesPerThread()) + "];\n"
    "<%DEFS%>"
    "\n"
    "// Load local memory\n"
    "unsigned int inLocalMemory = get_local_id(0);\n"
    "unsigned int inGlobalMemory = get_local_id(0) + (get_group_id(0) * " + isa::utils::toString(integration * conf.getNrSamplesPerThread()) + ");\n"
    "while ( inLocalMemory < " + isa::utils::toString(integration * conf.getNrSamplesPerThread()) + " ) {\n"
    "buffer[inLocalMemory] = input[(dm * " + isa::utils::toString(observation.getNrSamplesPerPaddedSecond(padding / sizeof(T))) + ") + inGlobalMemory];\n"
    "inGlobalMemory += " + isa::utils::toString(conf.getNrSamplesPerBlock()) + ";\n"
    "inLocalMemory += " + isa::utils::toString(conf.getNrSamplesPerBlock()) + ";\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Reduce\n"
    "unsigned int threshold = " + isa::utils::toString(conf.getNrSamplesPerBlock() / 2) + ";\n"
    "for ( unsigned int sample = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( sample < threshold ) {\n"
    "<%REDUCE%>"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "if ( get_local_id(0) < " + isa::utils::toString(conf.getNrSamplesPerThread()) + " ) {\n";
  if ( dataName == "float" ) {
    *code += "output[(dm * " + isa::utils::toString(isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(T))) + ") + ((get_group_id(0) * " + isa::utils::toString(conf.getNrSamplesPerThread()) + ") + get_local_id(0))] = buffer[get_local_id(0) * " + isa::utils::toString(integration) + "] * " + isa::utils::toString(1.0f / integration) + "f;\n";
  } else if ( dataName == "double" ) {
    *code += "output[(dm * " + isa::utils::toString(isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(T))) + ") + ((get_group_id(0) * " + isa::utils::toString(conf.getNrSamplesPerThread()) + ") + get_local_id(0))] = buffer[get_local_id(0) * " + isa::utils::toString(integration) + "] * " + isa::utils::toString(1.0 / integration) + ";\n";
  } else {
    *code += "output[(dm * " + isa::utils::toString(isa::utils::pad(observation.getNrSamplesPerSecond() / integration, padding / sizeof(T))) + ") + ((get_group_id(0) * " + isa::utils::toString(conf.getNrSamplesPerThread()) + ") + get_local_id(0))] = buffer[get_local_id(0) * " + isa::utils::toString(integration) + "] / " + isa::utils::toString(integration) + ";\n";
  }
  *code += "<%STORE%>"
    "}\n"
    "}\n";
  std::string defs_sTemplate = dataName + " integratedSample<%NUM%> = 0;\n";
  std::string reduce_sTemplate = "integratedSample<%NUM%> += buffer[(sample + <%OFFSET%>) + threshold];\n"
    "buffer[sample + <%OFFSET%>] = integratedSample<%NUM%>;\n";
  // End kernel's template

  std::string * defs_s = new std::string();
  std::string * reduce_s = new std::string();

  for ( unsigned int sample = 0; sample < conf.getNrSamplesPerThread(); sample++ ) {
    std::string sample_s = isa::utils::toString(sample);
    std::string offset_s = isa::utils::toString(sample * integration);
    std::string * temp = 0;

    temp = isa::utils::replace(&defs_sTemplate, "<%NUM%>", sample_s);
    defs_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&reduce_sTemplate, "<%NUM%>", sample_s);
    if ( sample == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    reduce_s->append(*temp);
    delete temp;
  }
  code = isa::utils::replace(code, "<%DEFS%>", *defs_s, true);
  code = isa::utils::replace(code, "<%REDUCE%>", *reduce_s, true);
  delete defs_s;
  delete reduce_s;

  return code;
}

} // PulsarSearch

#endif // INTEGRATION_HPP

