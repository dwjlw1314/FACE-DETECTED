/*
 * RetinaFace.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: ai_002
 */

#include "RetinaFace.h"

RetinaFace::RetinaFace(const char* path, MtcnnPar *param, bool useGpu)
{
	// TODO Auto-generated constructor stub
	m_executor = nullptr;
	m_nms_threshold = param->nms_threshold[0];
	if (useGpu)
		setContext(param->devid, mxnet::cpp::kGPU);
	else
		setContext(param->devid, mxnet::cpp::kCPU);

	vector<int> feat_stride_fpn = {32, 16, 8};

	map<int, AnchorCfg> anchor_cfg;
	anchor_cfg[32] = {std::vector<float>{32,16}, std::vector<float>{1}, 16};
	anchor_cfg[16] = {std::vector<float>{8,4}, std::vector<float>{1}, 16};
	anchor_cfg[8] = {std::vector<float>{2,1}, std::vector<float>{1}, 16};

	m_ag = vector<AnchorGenerator>(feat_stride_fpn.size());
	for (size_t i = 0; i < feat_stride_fpn.size(); ++i)
	{
		int stride = feat_stride_fpn[i];
		m_ag[i].Init(stride, param->threshold[1], anchor_cfg[stride], false);
	}
	InitModel(path);
}

RetinaFace::~RetinaFace()
{
	// TODO Auto-generated destructor stub
	delete m_ctx;
	//m_anchor_cfg.clear();
}

void RetinaFace::setContext(int id, DeviceType type)
{
	m_deviceId = id;
	m_deviceType = type;
	m_ctx = new Context(type, id);
}

void RetinaFace::InitModel(std::string floder)
{
	m_net = Symbol::Load(floder + "/mnet.25-symbol.json");
	std::map<std::string, mxnet::cpp::NDArray> params;
	NDArray::Load(floder + "/mnet.25-0000.params", nullptr, &params);
	for (const auto &k : params)
	{
		if (k.first.substr(0, 4) == "aux:")
		{
			auto name = k.first.substr(4, k.first.size() - 4);
			m_aux_map[name] = k.second.Copy(*m_ctx);
		}
		if (k.first.substr(0, 4) == "arg:")
		{
			auto name = k.first.substr(4, k.first.size() - 4);
			m_args_map[name] = k.second.Copy(*m_ctx);
		}
	}

	// WaitAll is need when we copy data between GPU and the main memory
	mxnet::cpp::NDArray::WaitAll();
	m_args_map["data"] = NDArray(Shape(1, 3, 45, 34), *m_ctx);
    m_executor = m_net.SimpleBind(*m_ctx, m_args_map, map<string, NDArray>(),
    		map<string, OpReqType>(), m_aux_map);
}

void RetinaFace::nms_cpu(std::vector<Anchor>& boxes, std::vector<Anchor>& filterOutBoxes)
{
    filterOutBoxes.clear();

    if(boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for(unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    //descending sort
    sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for(unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx][0], boxes[tmp_i][0] );
            float inter_y1 = std::max( boxes[good_idx][1], boxes[tmp_i][1] );
            float inter_x2 = std::min( boxes[good_idx][2], boxes[tmp_i][2] );
            float inter_y2 = std::min( boxes[good_idx][3], boxes[tmp_i][3] );

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
            float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);
            if(o <= m_nms_threshold)
                idx.push_back(tmp_i);
        }
    }
}

void RetinaFace::WrapInputLayer(std::vector<cv::Mat>* input_channels, const NDArray* input_layer,
                           const int height, const int width)
{
    mx_float *input_data = const_cast<mx_float*>(input_layer->GetData());
    for (mx_uint i = 0; i < input_layer->GetShape()[1]; ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void RetinaFace::Forward(NDArray* input)
{
    if (input)
    {
        if(m_executor)
            delete m_executor;
        if (input->GetContext().GetDeviceType() == m_deviceType)
        {
            m_args_map["data"] = *input;
        }
        else
        {
            m_args_map["data"] = input->Copy(*m_ctx);
        }
        m_executor = m_net.SimpleBind(*m_ctx, m_args_map, map<string, NDArray>(),
        		map<string, OpReqType>(), m_aux_map);
    }
    m_executor->Forward(false);
}

void RetinaFace::Detect(const cv::Mat& image, std::vector<FaceRect>& faceInfo)
{
    cv::Mat sample_single;
    //invert to RGB color space and float type
    image.convertTo(sample_single,CV_32FC3);
    //image.convertTo(sample_single, CV_32FC3, 0.0078125, -127.5 * 0.0078125);

    std::vector<Anchor> proposals;

	int idx = 0;
    int height = sample_single.rows;
    int width  = sample_single.cols;

    std::vector<cv::Mat> input_channels;
    // input data only using cpu, after Forward function convert
    NDArray input_layer(Shape(1, 3, height, width), Context(kCPU, 0));

    WrapInputLayer(&input_channels, &input_layer, height, width);
    cv::split(sample_single, input_channels);

    // check data transform right
    Forward(&input_layer);

	proposals.clear();
	for (int i = 0; i < 3; ++i)
	{
		idx = i * 3;
		//score
		NDArray score_data = m_executor->outputs[idx++].Copy(Context(kCPU, 0));
		NDArray bbox_data = m_executor->outputs[idx++].Copy(Context(kCPU, 0));
		NDArray landmark_deltas = m_executor->outputs[idx].Copy(Context(kCPU, 0));
		NDArray::WaitAll();

		m_ag[i].FilterAnchor(score_data, bbox_data, landmark_deltas, proposals);
	}

	// nms
	std::vector<Anchor> final_result;
	nms_cpu(proposals, final_result);

	for(Anchor& node : final_result)
	{
		FaceRect face;
		face.x1 = node.finalbox.x;
		face.x2 = node.finalbox.width;
		face.y1 = node.finalbox.y;
		face.y2 = node.finalbox.height;
		face.score = node.score;
		for(size_t i = 0; i < node.pts.size(); i++)
		{
			face.facepts.x[i] = node.pts[i].x;
			face.facepts.y[i] = node.pts[i].y;
		}
		faceInfo.push_back(face);
	}
}
