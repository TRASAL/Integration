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

class integrationConf {
public:
  integrationConf();
  ~integrationConf();
  // Get
  unsigned int getNrThreadsD0() const;
  unsigned int getNrItemsD0() const;
  // Set
  void setNrThreadsD0(unsigned int threads);
  void setNrItemsD0(unsigned int items);
  // utils
  std::string print() const;
private:
  unsigned int nrThreadsD0;
  unsigned int nrItemsD0;
};

typedef std::map< std::string, std::map < unsigned int, std::map< unsigned int, PulsarSearch::integrationConf * > * > * > tunedIntegrationConf;

// Sequential
template< typename T > void integrationDMsSamples(const AstroData::Observation & observation, const unsigned int integration, const unsigned int padding, const std::vector< T > & input, std::vector< T > & output);
template< typename T > void integrationSamplesDMs(const AstroData::Observation & observation, const unsigned int integration, const unsigned int padding, const std::vector< T > & input, std::vector< T > & output);
// OpenCL
template< typename T > std::string * getIntegrationDMsSamplesOpenCL(const integrationConf & conf, const unsigned int nrSamples, const std::string & inputDataName, const unsigned int integration, const unsigned int padding);
template< typename T > std::string * getIntegrationSamplesDMsOpenCL(const integrationConf & conf, const AstroData::Observation & observation, const std::string & inputDataName, const unsigned int integration, const unsigned int padding);
// Read configuration files
void readTunedIntegrationConf(tunedIntegrationConf & tunedConf, const std::string & confFilename);


// Implementations
inline unsigned int integrationConf::getNrThreadsD0() const {
  return nrThreadsD0;
}

inline unsigned int integrationConf::getNrItemsD0() const {
  return nrItemsD0;
}

inline void integrationConf::setNrThreadsD0(unsigned int threads) {
  nrThreadsD0 = threads;
}

inline void integrationConf::setNrItemsD0(unsigned int items) {
  nrItemsD0 = items;
}

template< typename T > void integrationDMsSamples(const AstroData::Observation & observation, const unsigned int integration, const unsigned int padding, const std::vector< T > & input, std::vector< T > & output) {
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

template< typename T > void integrationSamplesDMs(const AstroData::Observation & observation, const unsigned int integration, const unsigned int padding, const std::vector< T > & input, std::vector< T > & output) {
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample += integration ) {
      T integratedSample = 0;

      for ( unsigned int i = 0; i < integration; i++ ) {
        integratedSample += input[((sample + i) * observation.getNrPaddedDMs(padding / sizeof(T))) + dm];
      }
      output[((sample / integration) * observation.getNrPaddedDMs(padding / sizeof(T))) + dm] = integratedSample / integration;
    }
  }
}

template< typename T > std::string * getIntegrationDMsSamplesOpenCL(const integrationConf & conf, const unsigned int nrSamples, const std::string & dataName, const unsigned int integration, const unsigned int padding) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void integrationDMsSamples" + isa::utils::toString(integration) + "(__global const " + dataName + " * const restrict input, __global " + dataName + " * const restrict output) {\n"
    "unsigned int dm = get_group_id(1);\n"
    "__local " + dataName + " buffer[" + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + "];\n"
    "unsigned int inGlobalMemory = (dm * " + isa::utils::toString(isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (get_group_id(0) * " + isa::utils::toString(integration * conf.getNrItemsD0()) + ");\n"
    "<%DEFS%>"
    "\n"
    "// First computing phase\n"
    "for ( unsigned int sample = get_local_id(0); sample < " + isa::utils::toString(integration) + "; sample += " + isa::utils::toString(conf.getNrThreadsD0()) + " ) {\n"
    "<%SUM%>"
    "}\n"
    "<%LOAD%>"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Reduce\n"
    "unsigned int threshold = " + isa::utils::toString(conf.getNrThreadsD0() / 2) + ";\n"
    "for ( unsigned int sample = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( sample < threshold ) {\n"
    "<%REDUCE%>"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "inGlobalMemory = (dm * " + isa::utils::toString(isa::utils::pad(nrSamples / integration, padding / sizeof(T))) + ") + (get_group_id(0) * " + isa::utils::toString(conf.getNrItemsD0()) + ");\n"
    "if ( get_local_id(0) < " + isa::utils::toString(conf.getNrItemsD0()) + " ) {\n";
  if ( dataName == "float" ) {
    *code += "output[inGlobalMemory + get_local_id(0)] = buffer[get_local_id(0) * " + isa::utils::toString(conf.getNrThreadsD0()) + "] * " + isa::utils::toString(1.0f / integration) + "f;\n";
  } else if ( dataName == "double" ) {
    *code += "output[inGlobalMemory + get_local_id(0)] = buffer[get_local_id(0) * " + isa::utils::toString(conf.getNrThreadsD0()) + "] * " + isa::utils::toString(1.0 / integration) + ";\n";
  } else {
    *code += "output[inGlobalMemory + get_local_id(0)] = buffer[get_local_id(0) * " + isa::utils::toString(conf.getNrThreadsD0()) + "] / " + isa::utils::toString(integration) + ";\n";
  }
  *code += "}\n"
    "}\n";
  std::string defs_sTemplate = dataName + " integratedSample<%NUM%> = 0;\n";
  std::string sum_sTemplate = "integratedSample<%NUM%> += input[inGlobalMemory + sample + <%OFFSET%>];\n";
  std::string load_sTemplate = "buffer[get_local_id(0) + <%OFFSET%>] = integratedSample<%NUM%>;\n";
  std::string reduce_sTemplate = "integratedSample<%NUM%> += buffer[(sample + <%OFFSET%>) + threshold];\n"
    "buffer[sample + <%OFFSET%>] = integratedSample<%NUM%>;\n";
  // End kernel's template

  std::string * defs_s = new std::string();
  std::string * sum_s = new std::string();
  std::string * load_s = new std::string();
  std::string * reduce_s = new std::string();

  for ( unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++ ) {
    std::string sample_s = isa::utils::toString(sample);
    std::string offset_s = isa::utils::toString(sample * integration);
    std::string localOffset_s = isa::utils::toString(sample * conf.getNrThreadsD0());
    std::string * temp = 0;

    temp = isa::utils::replace(&defs_sTemplate, "<%NUM%>", sample_s);
    defs_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&sum_sTemplate, "<%NUM%>", sample_s);
    if ( sample == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    sum_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&load_sTemplate, "<%NUM%>", sample_s);
    if ( sample == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", localOffset_s, true);
    }
    load_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&reduce_sTemplate, "<%NUM%>", sample_s);
    if ( sample == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", localOffset_s, true);
    }
    reduce_s->append(*temp);
    delete temp;
  }
  code = isa::utils::replace(code, "<%DEFS%>", *defs_s, true);
  code = isa::utils::replace(code, "<%SUM%>", *sum_s, true);
  code = isa::utils::replace(code, "<%LOAD%>", *load_s, true);
  code = isa::utils::replace(code, "<%REDUCE%>", *reduce_s, true);
  delete defs_s;
  delete sum_s;
  delete load_s;
  delete reduce_s;

  return code;
}

template< typename T > std::string * getIntegrationSamplesDMsOpenCL(const integrationConf & conf, const AstroData::Observation & observation, const std::string & dataName, const unsigned int integration, const unsigned int padding) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void integrationSamplesDMs" + isa::utils::toString(integration) + "(__global const " + dataName + " * const restrict input, __global " + dataName + " * const restrict output) {\n"
    "unsigned int firstSample = get_group_id(1) * " + isa::utils::toString(observation.getNrSamplesPerSecond() / integration) + ";\n"
    "unsigned int dm = (get_group_id(0) * " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n"
    "<%DEFS%>"
    "\n"
    "for ( unsigned int sample = firstSample; sample < firstSample + " + isa::utils::toString(integration) + "; sample++ ) {\n"
    "<%SUM%>"
    "}\n"
    "<%STORE%>"
    "}\n";
  std::string defs_sTemplate = dataName + " integratedSample<%NUM%> = 0;\n";
  std::string sum_sTemplate = "integratedSample<%NUM%> += input[(sample * " + isa::utils::toString(observation.getNrPaddedDMs(padding / sizeof(T))) + ") + (dm + <%OFFSET%>)];\n";
  std::string store_sTemplate;
  if ( dataName == "float" ) {
    store_sTemplate += "output[(get_group_id(1) * " + isa::utils::toString(observation.getNrPaddedDMs(padding / sizeof(T))) + ") + (dm + <%OFFSET%>)] = integratedSample<%NUM%> * " + isa::utils::toString(1.0f / integration) + "f;\n";
  } else if ( dataName == "double" ) {
    store_sTemplate += "output[(get_group_id(1) * " + isa::utils::toString(observation.getNrPaddedDMs(padding / sizeof(T))) + ") + (dm + <%OFFSET%>)] = integratedSample<%NUM%> * " + isa::utils::toString(1.0 / integration) + ";\n";
  } else {
    store_sTemplate += "output[(get_group_id(1) * " + isa::utils::toString(observation.getNrPaddedDMs(padding / sizeof(T))) + ") + (dm + <%OFFSET%>)] = integratedSample<%NUM%> / " + isa::utils::toString(integration) + ";\n";
  }
  // End kernel's template

  std::string * defs_s = new std::string();
  std::string * sum_s = new std::string();
  std::string * store_s = new std::string();

  for ( unsigned int dm = 0; dm < conf.getNrItemsD0(); dm++ ) {
    std::string dm_s = isa::utils::toString(dm);
    std::string offset_s = isa::utils::toString(dm * conf.getNrThreadsD0());
    std::string * temp = 0;

    temp = isa::utils::replace(&defs_sTemplate, "<%NUM%>", dm_s);
    defs_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&sum_sTemplate, "<%NUM%>", dm_s);
    if ( dm == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    sum_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&store_sTemplate, "<%NUM%>", dm_s);
    if ( dm == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    store_s->append(*temp);
    delete temp;
  }
  code = isa::utils::replace(code, "<%DEFS%>", *defs_s, true);
  code = isa::utils::replace(code, "<%SUM%>", *sum_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete defs_s;
  delete sum_s;
  delete store_s;

  return code;
}

} // PulsarSearch

#endif // INTEGRATION_HPP

