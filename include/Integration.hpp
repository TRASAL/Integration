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

#include <string>
#include <map>
#include <vector>
#include <fstream>

#include <OpenCLTypes.hpp>
#include <Kernel.hpp>
#include <Observation.hpp>
#include <Platform.hpp>
#include <utils.hpp>

#pragma once

namespace Integration
{

class integrationConf : public isa::OpenCL::KernelConf
{
  public:
    integrationConf();
    ~integrationConf();
    // Get
    bool getSubbandDedispersion() const;
    // Set
    void setSubbandDedispersion(bool subband);
    // utils
    std::string print() const;

  private:
    bool subbandDedispersion;
};

typedef std::map<std::string, std::map<unsigned int, std::map<unsigned int, Integration::integrationConf *> *> *> tunedIntegrationConf;

// Sequential
template<typename NumericType>
void integrationBeforeDedispersion(const AstroData::Observation &observation, const unsigned int integration, const unsigned int padding, const std::vector<NumericType> &input, std::vector<NumericType> &output);
template <typename T>
void integrationDMsSamples(const bool subbandDedispersion, const AstroData::Observation &observation, const unsigned int integration, const unsigned int padding, const std::vector<T> &input, std::vector<T> &output);
template <typename T>
void integrationSamplesDMs(const bool subbandDedispersion, const AstroData::Observation &observation, const unsigned int integration, const unsigned int padding, const std::vector<T> &input, std::vector<T> &output);
// OpenCL
template <typename T>
std::string *getIntegrationDMsSamplesOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &inputDataName, const unsigned int integration, const unsigned int padding);
template <typename T>
std::string *getIntegrationSamplesDMsOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &inputDataName, const unsigned int integration, const unsigned int padding);
template<typename NumericType>
std::string *getIntegrationBeforeDedispersionInPlaceOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &dataName, const unsigned int integration, const unsigned int padding);
template<typename NumericType>
std::string *getIntegrationAfterDedispersionInPlaceOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &dataName, const unsigned int integration, const unsigned int padding);
template<typename NumericType>
std::string *getIntegrationInPlaceOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &dataName, const unsigned int dimOneSize, const unsigned int dimZeroSize, const unsigned int integration, const unsigned int padding);
// Read configuration files
void readTunedIntegrationConf(tunedIntegrationConf &tunedConf, const std::string &confFilename);

// Implementations
inline bool integrationConf::getSubbandDedispersion() const
{
    return subbandDedispersion;
}

inline void integrationConf::setSubbandDedispersion(bool subband)
{
    subbandDedispersion = subband;
}

template<typename NumericType>
void integrationBeforeDedispersion(const AstroData::Observation &observation, const unsigned int integration, const unsigned int padding, const std::vector<NumericType> &input, std::vector<NumericType> &output)
{
    for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ )
    {
        for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ )
        {
            for ( unsigned int sample = 0; sample < observation.getNrSamplesPerDispersedBatch(); sample += integration )
            {
                NumericType integratedSample = 0;

                for ( unsigned int i = 0; i < integration; i++ )
                {
                    integratedSample += input[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(NumericType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, padding / sizeof(NumericType))) + (sample + i)];
                }
                output[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / integration, padding / sizeof(NumericType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / integration, padding / sizeof(NumericType))) + (sample / integration)] = integratedSample / integration;
            }
        }
    }
}

template <typename T>
void integrationDMsSamples(const bool subbandDedispersion, const AstroData::Observation &observation, const unsigned int integration, const unsigned int padding, const std::vector<T> &input, std::vector<T> &output)
{
    unsigned int nrDMs = 0;

    if (subbandDedispersion)
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }

    for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
    {
        for (unsigned int dm = 0; dm < nrDMs; dm++)
        {
            for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / observation.getDownsampling(); sample += integration)
            {
                T integratedSample = 0;

                for (unsigned int i = 0; i < integration; i++)
                {
                    integratedSample += input[(beam * nrDMs * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling(), padding / sizeof(T))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling(), padding / sizeof(T))) + (sample + i)];
                }
                output[(beam * nrDMs * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling() / integration, padding / sizeof(T))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling() / integration, padding / sizeof(T))) + (sample / integration)] = integratedSample / integration;
            }
        }
    }
}

template <typename T>
void integrationSamplesDMs(const bool subbandDedispersion, const AstroData::Observation &observation, const unsigned int integration, const unsigned int padding, const std::vector<T> &input, std::vector<T> &output)
{
    unsigned int nrDMs = 0;

    if (subbandDedispersion)
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
    {
        for (unsigned int dm = 0; dm < nrDMs; dm++)
        {
            for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample += integration)
            {
                T integratedSample = 0;

                for (unsigned int i = 0; i < integration; i++)
                {
                    integratedSample += input[(beam * observation.getNrSamplesPerBatch() * isa::utils::pad(nrDMs, padding / sizeof(T))) + ((sample + i) * isa::utils::pad(nrDMs, padding / sizeof(T))) + dm];
                }
                output[(beam * (observation.getNrSamplesPerBatch() / integration) * isa::utils::pad(nrDMs, padding / sizeof(T))) + ((sample / integration) * isa::utils::pad(nrDMs, padding / sizeof(T))) + dm] = integratedSample / integration;
            }
        }
    }
}

template <typename T>
std::string *getIntegrationDMsSamplesOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &dataName, const unsigned int integration, const unsigned int padding)
{
    unsigned int nrDMs = 0;
    std::string *code = new std::string();

    if (conf.getSubbandDedispersion())
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    // Begin kernel's template
    *code = "__kernel void integrationDMsSamples" + std::to_string(integration) + "(__global const " + dataName + " * const restrict input, __global " + dataName + " * const restrict output) {\n"
    + conf.getIntType() + " beam = get_group_id(2);\n"
    + conf.getIntType() + " dm = get_group_id(1);\n"
    "__local " + dataName + " buffer[" + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + "];\n"
    + conf.getIntType() + " inGlobalMemory = (beam * " + std::to_string(nrDMs * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling(), padding / sizeof(T))) + ") + (dm * " + std::to_string(isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling(), padding / sizeof(T))) + ") + (get_group_id(0) * " + std::to_string(integration * conf.getNrItemsD0()) + ");\n"
    "<%DEFS%>"
    "\n"
    "// First computing phase\n"
    "for ( " + conf.getIntType() + " sample = get_local_id(0); sample < " + std::to_string(integration) + "; sample += " + std::to_string(conf.getNrThreadsD0()) + " ) {\n"
    "<%SUM%>"
    "}\n"
    "<%LOAD%>"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Reduce\n"
    + conf.getIntType() + " threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
    "for ( " + conf.getIntType() + " sample = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( sample < threshold ) {\n"
    "<%REDUCE%>"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "inGlobalMemory = (beam * " + std::to_string(nrDMs * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling() / integration, padding / sizeof(T))) + ") + (dm * " + std::to_string(isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling() / integration, padding / sizeof(T))) + ") + (get_group_id(0) * " + std::to_string(conf.getNrItemsD0()) + ");\n"
    "if ( get_local_id(0) < " + std::to_string(conf.getNrItemsD0()) + " ) {\n";
    if (dataName == "float")
    {
        *code += "output[inGlobalMemory + get_local_id(0)] = buffer[get_local_id(0) * " + std::to_string(conf.getNrThreadsD0()) + "] * " + std::to_string(1.0f / integration) + "f;\n";
    }
    else if (dataName == "double")
    {
        *code += "output[inGlobalMemory + get_local_id(0)] = buffer[get_local_id(0) * " + std::to_string(conf.getNrThreadsD0()) + "] * " + std::to_string(1.0 / integration) + ";\n";
    }
    else
    {
        *code += "output[inGlobalMemory + get_local_id(0)] = buffer[get_local_id(0) * " + std::to_string(conf.getNrThreadsD0()) + "] / " + std::to_string(integration) + ";\n";
    }
    *code += "}\n"
             "}\n";
    std::string defs_sTemplate = dataName + " integratedSample<%NUM%> = 0;\n";
    std::string sum_sTemplate = "integratedSample<%NUM%> += input[inGlobalMemory + sample + <%OFFSET%>];\n";
    std::string load_sTemplate = "buffer[get_local_id(0) + <%OFFSET%>] = integratedSample<%NUM%>;\n";
    std::string reduce_sTemplate = "integratedSample<%NUM%> += buffer[(sample + <%OFFSET%>) + threshold];\n"
    "buffer[sample + <%OFFSET%>] = integratedSample<%NUM%>;\n";
    // End kernel's template

    std::string *defs_s = new std::string();
    std::string *sum_s = new std::string();
    std::string *load_s = new std::string();
    std::string *reduce_s = new std::string();

    for (unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++)
    {
        std::string sample_s = std::to_string(sample);
        std::string offset_s = std::to_string(sample * integration);
        std::string localOffset_s = std::to_string(sample * conf.getNrThreadsD0());
        std::string *temp = 0;

        temp = isa::utils::replace(&defs_sTemplate, "<%NUM%>", sample_s);
        defs_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&sum_sTemplate, "<%NUM%>", sample_s);
        if (sample == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        sum_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&load_sTemplate, "<%NUM%>", sample_s);
        if (sample == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", localOffset_s, true);
        }
        load_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&reduce_sTemplate, "<%NUM%>", sample_s);
        if (sample == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
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

template <typename T>
std::string *getIntegrationSamplesDMsOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &dataName, const unsigned int integration, const unsigned int padding)
{
    unsigned int nrDMs = 0;
    std::string *code = new std::string();

    if (conf.getSubbandDedispersion())
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    // Begin kernel's template
    *code = "__kernel void integrationSamplesDMs" + std::to_string(integration) + "(__global const " + dataName + " * const restrict input, __global " + dataName + " * const restrict output) {\n"
    + conf.getIntType() + " beam = get_group_id(2);\n"
    + conf.getIntType() + " firstSample = get_group_id(1) * " + std::to_string(integration) + ";\n"
    + conf.getIntType() + " dm = (get_group_id(0) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n"
    "<%DEFS%>"
    "\n"
    "for ( " + conf.getIntType() + " sample = firstSample; sample < firstSample + " + std::to_string(integration) + "; sample++ ) {\n"
    "<%SUM%>"
    "}\n"
    "<%STORE%>"
    "}\n";
    std::string defs_sTemplate = dataName + " integratedSample<%NUM%> = 0;\n";
    std::string sum_sTemplate = "integratedSample<%NUM%> += input[(beam * " + std::to_string(observation.getNrSamplesPerBatch() * isa::utils::pad(nrDMs, padding / sizeof(T))) + " ) + (sample * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (dm + <%OFFSET%>)];\n";
    std::string store_sTemplate;
    if (dataName == "float")
    {
        store_sTemplate += "output[(beam * " + std::to_string((observation.getNrSamplesPerBatch() / integration) * isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (dm + <%OFFSET%>)] = integratedSample<%NUM%> * " + std::to_string(1.0f / integration) + "f;\n";
    }
    else if (dataName == "double")
    {
        store_sTemplate += "output[(beam * " + std::to_string((observation.getNrSamplesPerBatch() / integration) * isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (dm + <%OFFSET%>)] = integratedSample<%NUM%> * " + std::to_string(1.0 / integration) + ";\n";
    }
    else
    {
        store_sTemplate += "output[(beam * " + std::to_string((observation.getNrSamplesPerBatch() / integration) * isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (dm + <%OFFSET%>)] = integratedSample<%NUM%> / " + std::to_string(integration) + ";\n";
    }
    // End kernel's template

    std::string *defs_s = new std::string();
    std::string *sum_s = new std::string();
    std::string *store_s = new std::string();

    for (unsigned int dm = 0; dm < conf.getNrItemsD0(); dm++)
    {
        std::string dm_s = std::to_string(dm);
        std::string offset_s = std::to_string(dm * conf.getNrThreadsD0());
        std::string *temp = 0;

        temp = isa::utils::replace(&defs_sTemplate, "<%NUM%>", dm_s);
        defs_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&sum_sTemplate, "<%NUM%>", dm_s);
        if (dm == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        sum_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&store_sTemplate, "<%NUM%>", dm_s);
        if (dm == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
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

template<typename NumericType>
std::string *getIntegrationBeforeDedispersionInPlaceOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &dataName, const unsigned int integration, const unsigned int padding)
{
    return getIntegrationInPlaceOpenCL<NumericType>(conf, observation, dataName, observation.getNrChannels(), observation.getNrSamplesPerDispersedBatch(conf.getSubbandDedispersion()), integration, padding);
}

template<typename NumericType>
std::string *getIntegrationAfterDedispersionInPlaceOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &dataName, const unsigned int integration, const unsigned int padding)
{
    unsigned int nrDMs = 0;

    if ( conf.getSubbandDedispersion() )
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    return getIntegrationInPlaceOpenCL<NumericType>(conf, observation, dataName, nrDMs, observation.getNrSamplesPerBatch() / observation.getDownsampling(), integration, padding);
}

template<typename NumericType>
std::string *getIntegrationInPlaceOpenCL(const integrationConf &conf, const AstroData::Observation &observation, const std::string &dataName, const unsigned int dimOneSize, const unsigned int dimZeroSize, const unsigned int integration, const unsigned int padding)
{
    std::string *code = new std::string();
    // Begin kernel's template
    *code = "__kernel void integration" + std::to_string(integration) + "(__global " + dataName + " * const restrict data) {\n"
    "__local " + dataName + " buffer[" + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0() * integration) + "];\n"
    "for ( " + conf.getIntType() + " chunk = 0; chunk < " + std::to_string(static_cast<unsigned int>(std::ceil(static_cast<float>(dimZeroSize) / (conf.getNrThreadsD0() * conf.getNrItemsD0() * integration)))) + "; chunk++ ) {\n"
    "// Load samples in local memory\n"
    "<%DEFS%>"
    + conf.getIntType() + " inGlobalMemory = (get_group_id(2) * " + std::to_string(dimOneSize * isa::utils::pad(dimZeroSize, padding / sizeof(NumericType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(dimZeroSize, padding / sizeof(NumericType))) + ") + (chunk * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0() * integration) + ");\n"
    "for ( " + conf.getIntType() + " item = get_local_id(0); (item < " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0() * integration) + ") && (item + (chunk * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0() * integration) + ") < " + std::to_string(dimZeroSize) + "); item += " + std::to_string(conf.getNrThreadsD0()) + " ) {\n"
    "buffer[item] = data[inGlobalMemory + item];\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Integrate samples\n"
    "for ( " + conf.getIntType() + " item = 0; item < " + std::to_string(integration) + "; item++ ) {\n"
    "<%SUMS%>"
    "}\n"
    "// Store integrated data\n"
    "inGlobalMemory = (get_group_id(2) * " + std::to_string(dimOneSize * isa::utils::pad(dimZeroSize, padding / sizeof(NumericType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(dimZeroSize, padding / sizeof(NumericType))) + ") + (chunk * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ");\n"
    "<%STORES%>"
    "}\n"
    "}\n";
    std::string defs_sTemplate = dataName + " integratedSample<%NUM%> = 0;\n";
    std::string sum_sTemplate = "integratedSample<%NUM%> += buffer[(get_local_id(0) * " + std::to_string(integration) + ") + <%OFFSET%> + item];\n";
    std::string store_sTemplate = "data[inGlobalMemory + get_local_id(0) + <%OFFSET%>] = integratedSample<%NUM%> / " + std::to_string(integration) + ";\n";
    // End kernel's template

    std::string *defs_s = new std::string();
    std::string *sum_s = new std::string();
    std::string *store_s = new std::string();

    for (unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++)
    {
        std::string sample_s = std::to_string(sample);
        std::string offset_s = std::to_string(sample * integration * conf.getNrThreadsD0());
        std::string *temp = 0;

        temp = isa::utils::replace(&defs_sTemplate, "<%NUM%>", sample_s);
        defs_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&sum_sTemplate, "<%NUM%>", sample_s);
        if (sample == 0)
        {
            temp = isa::utils::replace(temp, " + <%OFFSET%>", "", true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        sum_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&store_sTemplate, "<%NUM%>", sample_s);
        offset_s = std::to_string(sample * conf.getNrThreadsD0());
        if (sample == 0)
        {
            temp = isa::utils::replace(temp, " + <%OFFSET%>", "", true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        store_s->append(*temp);
        delete temp;
    }
    code = isa::utils::replace(code, "<%DEFS%>", *defs_s, true);
    code = isa::utils::replace(code, "<%SUMS%>", *sum_s, true);
    code = isa::utils::replace(code, "<%STORES%>", *store_s, true);
    delete defs_s;
    delete sum_s;
    delete store_s;

    return code;
}

} // namespace Integration
