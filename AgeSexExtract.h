/*
 * AgeSexExtract.h
 *
 *  Created on: Nov 26, 2019
 *      Author: ai_002
 */

#ifndef AGESEXEXTRACT_H_
#define AGESEXEXTRACT_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "mxnet-cpp/MxNetCpp.h"
#include "public_data.h"

using namespace std;
using namespace mxnet::cpp;
using namespace IODATA;

class AgeSexExtract
{
public:
	AgeSexExtract(const char*, MtcnnPar*, bool);

	~AgeSexExtract();

    void InitFaceGender(const string&);

    void SetParameter(const MtcnnPar*);

    void GetFaceAgeSex(std::vector<cv::Mat>&, pFaceRect);

private:
#if USE_MOREBATCH
    NDArray Mat2NDArray(std::vector<cv::Mat>&);
#else
    void Mat2NDArray(cv::Mat&);
#endif

    void setContext(int, DeviceType);

    void LoadParamtes(const string&);

    void SetGenderLayerName(string);

    void SetGenderShape(const int (&)[2]);

private:
	int m_deviceId;
	DeviceType m_deviceType;
    Context *m_ctx = nullptr;

    int m_GenderShape[2];
    string m_FlayerName;

    Symbol m_Gnet;
    Executor* m_Gender;

    map<string, NDArray> m_Args_Map;
    map<string, NDArray> m_Aux_Map;
};

#endif /* AGESEXEXTRACT_H_ */
