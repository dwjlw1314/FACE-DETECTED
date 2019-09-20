/*
 * RetinaFace.h
 *
 *  Created on: Sep 19, 2019
 *      Author: ai_002
 */

#ifndef RETINAFACE_H_
#define RETINAFACE_H_

/*
 * The switch of image value selection mode. can increase the running speed
 * and reduce the detection accuracy after it is turned on.
 */
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "mxnet-cpp/MxNetCpp.h"
#include "public_data.h"
#include "anchor_generator.h"

using namespace std;
using namespace mxnet::cpp;
using namespace IODATA;

class RetinaFace
{
public:
	RetinaFace(const char*, MtcnnPar *, bool);
	virtual ~RetinaFace();

	void Detect(const cv::Mat&, std::vector<FaceRect>&);

	// Init RetinaFace Model
	void InitModel(std::string);

	void setContext(int id = 0, DeviceType type = kCPU);

protected:
	void nms_cpu(vector<Anchor>&, vector<Anchor>&);
    void WrapInputLayer(vector<cv::Mat>*, const NDArray*, const int, const int);
	void Forward(NDArray*);

private:
	int m_deviceId;
	DeviceType m_deviceType;

    mx_float m_nms_threshold;

    Context *m_ctx = nullptr;

    Symbol m_net;
    Executor* m_executor;

    map<std::string, NDArray> m_args_map;
    map<std::string, NDArray> m_aux_map;

	std::vector<AnchorGenerator> m_ag;
};

#endif /* RETINAFACE_H_ */
