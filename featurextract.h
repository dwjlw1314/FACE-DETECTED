#ifndef FEATUREXTRACT_H
#define FEATUREXTRACT_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "mxnet-cpp/MxNetCpp.h"
#include "public_data.h"

#define USE_MOREBATCH 0

using namespace std;
using namespace mxnet::cpp;
using namespace IODATA;

class FeaturExtract
{
public:
    FeaturExtract(const char*, MtcnnPar*);

    ~FeaturExtract();

    void InitFaceFeature(const string&);

    void SetParameter(const MtcnnPar*);

    void GetFaceFeature(std::vector<cv::Mat>&, pFaceRect);

    mx_float getSimilarity(const mx_float*, const mx_float*);

private:
#if USE_MOREBATCH
    NDArray Mat2NDArray(std::vector<cv::Mat>&);
#else
    void Mat2NDArray(cv::Mat&);
#endif

    void LoadParamtes(const string&);

    mx_float getMold(const mx_float*);

    void SetFeatureLayerName(string);

    void SetFeatureShape(const int (&)[2]);

private:

    int m_FeatureShape[2];
    string m_FlayerName;

    Symbol m_Fnet;
    Executor* m_Fexecutor;

    map<string, NDArray> m_Args_Map;
    map<string, NDArray> m_Aux_Map;
};

#endif // FEATUREXTRACT_H
