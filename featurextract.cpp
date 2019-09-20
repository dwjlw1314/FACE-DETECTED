#include "featurextract.h"

extern Context g_GetContext();

FeaturExtract::FeaturExtract(const char* path, MtcnnPar *param)
{
	SetParameter(param);
	InitFaceFeature(path);
}

FeaturExtract::~FeaturExtract()
{
	delete m_Fexecutor;
}

inline void FeaturExtract::SetFeatureLayerName(string layer_name)
{
	m_FlayerName = layer_name;
}

inline void FeaturExtract::SetFeatureShape(const int (&array)[2])
{
	m_FeatureShape[0] = array[0];
	m_FeatureShape[1] = array[1];
}

void FeaturExtract::SetParameter(const MtcnnPar* param)
{
    SetFeatureShape(param->featureshape);
    SetFeatureLayerName(param->featurelayername);
}

void FeaturExtract::LoadParamtes(const string& model_name)
{
    map<string, NDArray> paramters;
    NDArray::Load(model_name, nullptr, &paramters);
    for (const auto &k : paramters)
    {
        if (k.first.substr(0, 4) == "aux:")
        {
            auto name = k.first.substr(4, k.first.size() - 4);
            m_Aux_Map[name] = k.second.Copy(g_GetContext());
        }
        if (k.first.substr(0, 4) == "arg:")
        {
            auto name = k.first.substr(4, k.first.size() - 4);
            m_Args_Map[name] = k.second.Copy(g_GetContext());
        }
    }
    /*WaitAll is need when we copy data between GPU and the main memory*/
    NDArray::WaitAll();
}

void FeaturExtract::InitFaceFeature(const string& path)
{
    m_Fnet = Symbol::Load(path+"/face/model-symbol.json").GetInternals()[m_FlayerName];
    LoadParamtes(path+"/face/model-0000.params");
    m_Args_Map["data"] = NDArray(Shape(1, 3, m_FeatureShape[0], m_FeatureShape[1]), g_GetContext());
    m_Fexecutor = m_Fnet.SimpleBind(g_GetContext(), m_Args_Map, map<string, NDArray>(),
    		map<string, OpReqType>(), m_Aux_Map);
}

#if USE_MOREBATCH
NDArray FeaturExtract::Mat2NDArray(std::vector<cv::Mat>& v_feature_mat)
{
	size_t count = v_feature_mat.size();
	size_t width = count * 3 * m_FeatureShape[0] * m_FeatureShape[1];

	NDArray ret(Shape(count, 3, m_FeatureShape[0], m_FeatureShape[1]), Context::cpu());
	std::vector<float> array;
	for(auto image : v_feature_mat)
	{
		if (image.cols != 112 || image.rows != 112)
			cv::resize(image, image, cv::Size(m_FeatureShape[0], m_FeatureShape[1]));

		for (int c = 0; c < 3; ++c)
		{
			for (int i = 0; i < m_FeatureShape[0]; ++i)
			{
				for (int j = 0; j < m_FeatureShape[1]; ++j)
				{
					array.push_back(static_cast<float>(image.data[(i * m_FeatureShape[0] + j) * 3 + c]));
				}
			}
		}
	}
	ret.SyncCopyFromCPU(array.data(), (size_t)width);
	NDArray::WaitAll();

    return ret;
}

void FeaturExtract::GetFaceFeature(std::vector<cv::Mat>& v_feature_mat, pFaceRect face_point)
{
	size_t count = v_feature_mat.size();
	m_Args_Map["data"] = NDArray(Shape(count, 3, m_FeatureShape[0], m_FeatureShape[1]), g_GetContext());
	Mat2NDArray(v_feature_mat).CopyTo(&m_Args_Map["data"]);
	NDArray::WaitAll();

	delete m_Fexecutor;
	m_Fexecutor = m_Fnet.SimpleBind(g_GetContext(), m_Args_Map, map<string, NDArray>(),
			map<string, OpReqType>(), m_Aux_Map);

	m_Fexecutor->Forward(false);

	/*out the features*/
	auto array = m_Fexecutor->outputs[0].Copy(Context::cpu());
	NDArray::WaitAll();

	for(size_t i = 0; i < count; i++)
	{
		for (size_t j = 0; j < FSIZE; j++)
		{
			face_point->feature[j] = array.At(0,i*FSIZE+j);
		}
		face_point++;
	}
}
#else
NDArray FeaturExtract::Mat2NDArray(cv::Mat& image)
{
    std::vector<float> array;

    if (image.cols != 112 || image.rows != 112)
    	cv::resize(image, image, cv::Size(m_FeatureShape[0], m_FeatureShape[1]));

    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < m_FeatureShape[0]; ++i)
        {
            for (int j = 0; j < m_FeatureShape[1]; ++j)
            {
                array.push_back(static_cast<float>(image.data[(i * m_FeatureShape[0] + j) * 3 + c]));
            }
        }
    }
    NDArray ret(Shape(1, 3, m_FeatureShape[0], m_FeatureShape[1]), Context::cpu());
    int length = 1 * 3 * m_FeatureShape[0] * m_FeatureShape[1];
    ret.SyncCopyFromCPU(array.data(), (size_t)length);
    NDArray::WaitAll();
    return ret;
}

void FeaturExtract::GetFaceFeature(std::vector<cv::Mat>& v_feature_mat, pFaceRect face_point)
{
    for (auto image : v_feature_mat)
    {
    	Mat2NDArray(image).CopyTo(&m_Args_Map["data"]);
        NDArray::WaitAll();

        m_Fexecutor->Forward(false);

        /*out the features*/
        auto array = m_Fexecutor->outputs[0].Copy(Context::cpu());
        NDArray::WaitAll();

        for (size_t i = 0; i < FSIZE; i++)
        {
            face_point->feature[i] = array.At(0,i);
        }
        face_point++;
    }
}
#endif

//求向量的模长
mx_float FeaturExtract::getMold(const mx_float* vec)
{
    mx_float sum = 0.0;
    for (int i = 0; i < FSIZE; ++i)
        sum += vec[i] * vec[i];
    return sqrt(sum);
}

//特征余弦相似度
mx_float FeaturExtract::getSimilarity(const mx_float* lhs, const mx_float* rhs)
{
     mx_float tmp = 0.0;  //内积
     for (int i = 0; i < FSIZE; ++i)
         tmp += lhs[i] * rhs[i];
     return tmp / (getMold(lhs)*getMold(rhs));
}

