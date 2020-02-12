/*
 * AgeSexExtract.cpp
 *
 *  Created on: Nov 26, 2019
 *      Author: ai_002
 */

#include "AgeSexExtract.h"

AgeSexExtract::AgeSexExtract(const char* path, MtcnnPar *param, bool useGpu)
	:m_Gender(nullptr)
{
	if (useGpu)
		setContext(param->devid, mxnet::cpp::kGPU);
	else
		setContext(param->devid, mxnet::cpp::kCPU);

	SetParameter(param);
	InitFaceGender(path);
}

AgeSexExtract::~AgeSexExtract()
{
	delete m_ctx;
	delete m_Gender;
}

void AgeSexExtract::setContext(int id, DeviceType type)
{
	m_deviceId = id;
	m_deviceType = type;
	m_ctx = new Context(type, id);
}

void AgeSexExtract::SetParameter(const MtcnnPar* param)
{
    SetGenderShape(param->gendershape);
    SetGenderLayerName(param->featurelayername);
}

inline void AgeSexExtract::SetGenderLayerName(string layer_name)
{
	m_FlayerName = layer_name;
}

inline void AgeSexExtract::SetGenderShape(const int (&array)[2])
{
	m_GenderShape[0] = array[0];
	m_GenderShape[1] = array[1];
}

//GPU increase 200M
void AgeSexExtract::LoadParamtes(const string& model_name)
{
    map<string, NDArray> paramters;
    NDArray::Load(model_name, nullptr, &paramters);
    for (const auto &k : paramters)
    {
        if (k.first.substr(0, 4) == "aux:")
        {
            auto name = k.first.substr(4, k.first.size() - 4);
            m_Aux_Map[name] = k.second.Copy(*m_ctx);
        }
        if (k.first.substr(0, 4) == "arg:")
        {
            auto name = k.first.substr(4, k.first.size() - 4);
            m_Args_Map[name] = k.second.Copy(*m_ctx);
        }
    }
    /*WaitAll is need when we copy data between GPU and the main memory*/
    NDArray::WaitAll();
}

void AgeSexExtract::InitFaceGender(const string& path)
{
    m_Gnet = Symbol::Load(path+"/gender/model-symbol.json").GetInternals()[m_FlayerName];
    LoadParamtes(path+"/gender/model-0000.params");
    m_Args_Map["data"] = NDArray(Shape(1, 3, m_GenderShape[0], m_GenderShape[1]), *m_ctx);
    //GPU increase 350M
    m_Gender = m_Gnet.SimpleBind(*m_ctx, m_Args_Map, map<string, NDArray>(),
    		map<string, OpReqType>(), m_Aux_Map);
}

#if USE_MOREBATCH
NDArray AgeSexExtract::Mat2NDArray(std::vector<cv::Mat>& v_feature_mat)
{
	size_t count = v_feature_mat.size();
	size_t width = count * 3 * m_GenderShape[0] * m_GenderShape[1];

	NDArray ret(Shape(count, 3, m_GenderShape[0], m_GenderShape[1]), Context::cpu());
	std::vector<float> array;
	for(auto image : v_feature_mat)
	{
		if (image.cols != m_GenderShape[0] || image.rows != m_GenderShape[1])
			cv::resize(image, image, cv::Size(m_GenderShape[0], m_GenderShape[1]));

		for (int c = 0; c < 3; ++c)
		{
			for (int i = 0; i < m_GenderShape[0]; ++i)
			{
				for (int j = 0; j < m_GenderShape[1]; ++j)
				{
					array.push_back(static_cast<float>(image.data[(i * m_GenderShape[0] + j) * 3 + c]));
				}
			}
		}
	}
	ret.SyncCopyFromCPU(array.data(), (size_t)width);
	NDArray::WaitAll();

    return ret;
}

void AgeSexExtract::GetFaceAgeSex(std::vector<cv::Mat>& v_feature_mat, pFaceRect face_point)
{
	size_t count = v_feature_mat.size();
	m_Args_Map["data"] = NDArray(Shape(count, 3, m_GenderShape[0], m_GenderShape[1]), *m_ctx);
	Mat2NDArray(v_feature_mat).CopyTo(&m_Args_Map["data"]);

	delete m_Gender;
	m_Gender = m_Gnet.SimpleBind(*m_ctx, m_Args_Map, map<string, NDArray>(),
			map<string, OpReqType>(), m_Aux_Map);

	m_Gender->Forward(false);

	/*out the features*/
	NDArray array = m_Gender->outputs[0].Copy(Context::cpu());
	NDArray::WaitAll();

	const mx_float *out_data = array.GetData();
	size_t out_data_size = array.Size();

	for(size_t i = 0; i < count; i++)
	{
		vector<int> ages;
		int offset = i * 202;
		mx_float a = out_data[0 + offset];
		mx_float b = out_data[1 + offset];

		face_point->gender = b > a ? 1 : 2;

		vector<int> ages;
		for (size_t j = 1; j <= 100; j++)
		{
			ages.push_back(out_data[2 * j + offset] > out_data[2 * j + 1 + offset] ? 0 : 1);
		}
		face_point->age = accumulate(ages.begin(), ages.end(), 0);

		face_point++;
	}
}
#else
void AgeSexExtract::Mat2NDArray(cv::Mat& image)
{
    std::vector<float> array;

    if (image.cols != m_GenderShape[0] || image.rows != m_GenderShape[1])
    	cv::resize(image, image, cv::Size(m_GenderShape[0], m_GenderShape[1]));

    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < m_GenderShape[0]; ++i)
        {
            for (int j = 0; j < m_GenderShape[1]; ++j)
            {
                array.push_back(static_cast<float>(image.data[(i * m_GenderShape[0] + j) * 3 + c]));
            }
        }
    }
    NDArray ret(Shape(1, 3, m_GenderShape[0], m_GenderShape[1]), Context::cpu());
    int length = 1 * 3 * m_GenderShape[0] * m_GenderShape[1];
    ret.SyncCopyFromCPU(array.data(), (size_t)length);
    ret.CopyTo(&m_Args_Map["data"]);
	NDArray::WaitAll();  //ret.WaitToRead();
}

void AgeSexExtract::GetFaceAgeSex(std::vector<cv::Mat>& v_feature_mat, pFaceRect face_point)
{
    for (auto image : v_feature_mat)
    {
    	try
    	{
			Mat2NDArray(image);

			m_Gender->Forward(false);

			/*out the features*/
//			std::vector<mx_float> output_data;
//			m_Gender->outputs[0].SyncCopyToCPU(&output_data, m_Gender->outputs[0].Size());
			auto array = m_Gender->outputs[0].Copy(Context::cpu());
			NDArray::WaitAll();

			const mx_float *out_data = array.GetData();
			size_t out_data_size = (array.Size() - 2)/2;

			face_point->gender = out_data[1] > out_data[0] ? 0 : 1;

//			cout << "first: " << out_data[0] << "second: " << out_data[1] << endl;

			vector<int> ages;
			for(size_t j = 1; j <= out_data_size; j++)
				ages.push_back(out_data[2 * j] > out_data[2 * j + 1] ? 0 : 1);

			face_point->age = accumulate(ages.begin(), ages.end(), 0);

			/* PYTHON code
			ret = self.ga_model.get_outputs()[0].asnumpy()
			g = ret[:, 0:2].flatten()
			gender = np.argmax(g)
			a = ret[:, 2:202].reshape((100, 2))
			a = np.argmax(a, axis=1)
			age = int(sum(a))
			*/
    	}
    	catch(...)
    	{
    		face_point->age = 0.0;
    		face_point->gender = -1;
    	}
    	face_point++;
    }
}
#endif
