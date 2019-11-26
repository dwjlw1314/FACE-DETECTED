#include "mtcnn_net.h"

extern Context g_GetContext();
extern DeviceType g_DeviceType;

NetStruct::NetStruct() : m_executor(nullptr)
{
	m_threshold = 0.0;
	m_nms_threshold = 0.0;
}

NetStruct::~NetStruct()
{
	DelExecutorNDarray();
    delete m_executor;
}

inline void NetStruct::SetThreshold(const mx_float value)
{
	m_threshold = value;
}

inline mx_float NetStruct::GetThreshold() const
{
	return m_threshold;
}

inline void NetStruct::SetNmsThreshold(const mx_float value)
{
	m_nms_threshold = value;
}

inline mx_float NetStruct::GetNmsThreshold() const
{
	return m_nms_threshold;
}

inline NDArray& NetStruct::InputDataNDArray()
{
    return m_args_map["data"];
}

inline const NDArray* NetStruct::OutputNDArray(size_t index)
{
    return m_out_ndarray[index];
}

inline const std::vector<mx_uint> NetStruct::GetDataShape()
{
    return m_args_map["data"].GetShape();
}

void NetStruct::InitMember(std::string symbol_name, std::string param_name, Shape shape)
{
    if (shape[2] && shape[3])
    {
        m_net = Symbol::Load(symbol_name);
        LoadParameters(param_name);
        m_args_map["data"] = NDArray(shape, g_GetContext());
        m_executor = m_net.SimpleBind(g_GetContext(), m_args_map);
    }
}

void NetStruct::Forward(NDArray* input)
{
    if (input)
    {
        if(m_executor)
            delete m_executor;
        if (input->GetContext().GetDeviceType() == g_DeviceType)
        {
            m_args_map["data"] = *input;
        }
        else
        {
            NDArray data(input->GetShape(), g_GetContext());
            input->CopyTo(&data);
            m_args_map["data"] = data;
        }
        m_executor = m_net.SimpleBind(g_GetContext(), m_args_map);
    }
    m_executor->Forward(false);
    GetExecutorNDarray();
}

void NetStruct::LoadParameters(const string& file)
{
    map<string, NDArray> paramters;
    NDArray::Load(file, nullptr, &paramters);

    for (const auto &k : paramters)
    {
        if (k.first.substr(0, 4) == "aux:")
        {
            auto name = k.first.substr(4, k.first.size() - 4);
            m_aux_map[name] = k.second.Copy(g_GetContext());
        }
        if (k.first.substr(0, 4) == "arg:")
        {
            auto name = k.first.substr(4, k.first.size() - 4);
            m_args_map[name] = k.second.Copy(g_GetContext());
        }
    }

    /*WaitAll is need when we copy data between GPU and the main memory*/
    NDArray::WaitAll();
}

void NetStruct::GetExecutorNDarray()
{
    if (!m_out_ndarray.empty())
        DelExecutorNDarray();
    for (auto& output : m_executor->outputs)
    {
        auto data = new NDArray(output.GetShape(), Context::cpu());
        output.SyncCopyToCPU(const_cast<mx_float*>(data->GetData()), data->Size());
        m_out_ndarray.push_back(data);
    }
}

void NetStruct::DelExecutorNDarray()
{
    for (size_t i = 0; i < m_out_ndarray.size() ; i++)
        delete m_out_ndarray[i];
    m_out_ndarray.erase(m_out_ndarray.begin(), m_out_ndarray.end());
}

/*
 * ************************************ MTCNN **********************************
 *
 * MTCC,Multi-task convolutional neural network（多任务卷积神经网络），基于cascade框架。
 * MTCNN由3个网络结构组成（P-Net,R-Net,O-Net）Proposal Network
 * 卷积神经网络里面最重要也是最基本的概念就是卷积层、池化层、全连接层、卷积核、参数共享
 */
//compare score
bool ComparefaceBox(const FaceRect& a, const FaceRect& b)
{
    return a.score > b.score;
}

MTCNN::MTCNN(const char* path, MtcnnPar *param, bool useGpu) :
	m_PNet(new NetStruct),
	m_RNet(new NetStruct),
	m_ONet(new NetStruct)
{
	m_factor = 0.0;
	m_minSize = 0;
	SetParameter(param);
	InitMtcnn(path);
}

MTCNN::~MTCNN()
{
	delete m_PNet;
	delete m_RNet;
	delete m_ONet;
}

void MTCNN::InitMtcnn(const string& proto_model_dir)
{
    /* Load face frame the network. */
    m_PNet->InitMember(proto_model_dir+"det1-symbol.json", proto_model_dir+"det1-0001.params", Shape(1, 3, 12, 12));
    m_RNet->InitMember(proto_model_dir+"det2-symbol.json", proto_model_dir+"det2-0001.params", Shape(1, 3, 24, 24));
    m_ONet->InitMember(proto_model_dir+"det3-symbol.json", proto_model_dir+"det3-0001.params", Shape(1, 3, 48, 48));
}

void MTCNN::SetParameter(const MtcnnPar* param)
{
    m_minSize = param->minSize;
    m_factor = param->factor;
    m_PNet->SetThreshold(param->threshold[0]);
    m_PNet->SetNmsThreshold(param->nms_threshold[0]);
    m_RNet->SetThreshold(param->threshold[1]);
    m_RNet->SetNmsThreshold(param->nms_threshold[1]);
    m_ONet->SetThreshold(param->threshold[2]);
    m_ONet->SetNmsThreshold(param->nms_threshold[2]);
}

void MTCNN::Detect(const cv::Mat& image, std::vector<FaceRect>& faceInfo)
{
    //2~3ms
    cv::Mat sample_single,resized;
    //invert to RGB color space and float type
    image.convertTo(sample_single,CV_32FC3);
    //cvtColor(sample_single, sample_single, COLOR_BGR2RGB);

    int height = image.rows;
    int width  = image.cols;
    int minWH = std::min(height, width);
    size_t factor_count = 0;
    double scale = 12.0/m_minSize;
    minWH *= scale;
    std::vector<double> scales;
    while (minWH >= 12)
    {
        scales.push_back(scale);
        minWH *= m_factor;
        scale *= m_factor;
        factor_count++;
    }

    // 11ms main consum
    for (std::vector<double>::size_type i = 0; i < factor_count; i++)
    {
        double scale = scales[i];
        double hs = ceil(height*scale);
        double ws = ceil(width*scale);

        std::vector<cv::Mat> input_channels;
        //wrap image and normalization using INTER_AREA method
#ifdef INTER_FAST
        cv::resize(sample_single,resized, cv::Size(ws,hs), 0, 0, cv::INTER_NEAREST);
#else
        cv::resize(sample_single,resized, cv::Size(ws,hs), 0, 0, cv::INTER_AREA);
#endif
        resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5 * 0.0078125);

        // input data only using cpu, after Forward function convert
        NDArray input_layer(Shape(1, 3, hs, ws), Context::cpu());

        WrapInputLayer(&input_channels, &input_layer, hs, ws);
        cv::split(resized, input_channels);

        // check data transform right
        m_PNet->Forward(&input_layer);

        const NDArray* reg = m_PNet->OutputNDArray(0);
        const NDArray* confidence = m_PNet->OutputNDArray(1);
        GenerateBoundingBox(confidence, reg, scale, m_PNet->GetThreshold(), ws, hs);

        std::vector<FaceRect> bboxes_nms = NonMaximumSuppression(m_condidate_rects, m_PNet->GetNmsThreshold(), 'u');
        m_total_boxes.insert(m_total_boxes.end(), bboxes_nms.begin(), bboxes_nms.end());
    }

    size_t numBox = m_total_boxes.size();
    if (numBox != 0)
    {
        m_total_boxes = NonMaximumSuppression(m_total_boxes, m_RNet->GetNmsThreshold(), 'u');
        m_regressed_rects = BoxRegress(m_total_boxes, 1);
        m_total_boxes.clear();

        Bbox2Square(m_regressed_rects);
        Padding(width, height);

        //RNet
        if (g_DeviceType == mxnet::cpp::kCPU)
            ClassifyFace(m_regressed_rects, sample_single, m_RNet, m_RNet->GetThreshold(), 'r');
        else
            ClassifyFace_MulImage(m_regressed_rects, sample_single, m_RNet, m_RNet->GetThreshold(), 'r');

        m_condidate_rects = NonMaximumSuppression(m_condidate_rects, m_RNet->GetNmsThreshold(), 'u');
        m_regressed_rects = BoxRegress(m_condidate_rects, 2);

        Bbox2Square(m_regressed_rects);
        Padding(width, height);

        //ONet
        numBox = m_regressed_rects.size();
        if(numBox != 0)
        {
            if (g_DeviceType == mxnet::cpp::kCPU)
                ClassifyFace(m_regressed_rects, sample_single, m_ONet, m_ONet->GetThreshold(), 'o');
            else
                ClassifyFace_MulImage(m_regressed_rects, sample_single,m_ONet, m_ONet->GetThreshold(), 'o');

            m_regressed_rects = BoxRegress(m_condidate_rects, 3);
            faceInfo = NonMaximumSuppression(m_regressed_rects, m_ONet->GetNmsThreshold(), 'm');
        }
    }
    m_regressed_pading.clear();
    m_regressed_rects.clear();
    m_condidate_rects.clear();
}

void MTCNN::WrapInputLayer(std::vector<cv::Mat>* input_channels, const NDArray* input_layer,
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

void MTCNN::GenerateBoundingBox(const NDArray* confidence, const NDArray* reg, double scale, const mx_float thresh,
                                int image_width, int image_height)
{
    int stride = 2;
    int cellSize = 12;

    size_t curr_feature_map_w = std::ceil((image_width - cellSize) * 1.0 / stride) + 1;
    size_t curr_feature_map_h = std::ceil((image_height - cellSize) * 1.0 / stride) + 1;

    //std::cout << "Feature_map_size:"<< curr_feature_map_w_ <<" "<<curr_feature_map_h_<<std::endl;
    size_t regOffset = curr_feature_map_w*curr_feature_map_h;

    // the first count numbers are confidence of face
    size_t count = confidence->Size()/2;
    const mx_float* confidence_data = confidence->GetData();
    confidence_data += count;
    const float* reg_data = reg->GetData();

    m_condidate_rects.clear();
    for (size_t i = 0; i < count; i++)
    {
        if (*(confidence_data + i) >= thresh)
        {
        	FaceRect faceInfo;
            int y = i / curr_feature_map_w;
            int x = i - curr_feature_map_w * y;

            faceInfo.x1 = static_cast<int>(((x * stride + 1)/scale));
            faceInfo.y1 = static_cast<int>(((y * stride + 1)/scale));
            faceInfo.x2 = static_cast<int>(((x * stride + cellSize-1 + 1) / scale));
            faceInfo.y2 = static_cast<int>(((y * stride + cellSize-1 + 1) / scale));
            faceInfo.score  = *(confidence_data+i);

            faceInfo.regression[0] = reg_data[i + 0 * regOffset];
            faceInfo.regression[1] = reg_data[i + 1 * regOffset];
            faceInfo.regression[2] = reg_data[i + 2 * regOffset];
            faceInfo.regression[3] = reg_data[i + 3 * regOffset];

            m_condidate_rects.push_back(faceInfo);
        }
    }
}

// methodType : u is IoU(Intersection Over Union)
// methodType : m is IoM(Intersection Over Maximum)
std::vector<FaceRect> MTCNN::NonMaximumSuppression(std::vector<FaceRect>& faceBoxes, const mx_float thresh, char IOUType)
{
    std::vector<FaceRect> bboxes_nms;
    std::sort(faceBoxes.begin(), faceBoxes.end(), ComparefaceBox);

    size_t select_idx = 0;
    size_t num_bbox = faceBoxes.size();
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged)
    {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox)
        {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(faceBoxes[select_idx]);
        mask_merged[select_idx] = 1;

        FaceRect& select_bbox = faceBoxes[select_idx];
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (size_t i = select_idx; i < num_bbox; i++)
        {
            if (mask_merged[i] == 1)
                continue;

            FaceRect& bbox_i = faceBoxes[i];
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;

            switch (IOUType) {
            case 'u':
                if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
                    mask_merged[i] = 1;
                break;
            case 'm':
                if (static_cast<float>(area_intersect) / std::min(area1 , area2) > thresh)
                    mask_merged[i] = 1;
                break;
            default:
                break;
            }
        }
    }
    return bboxes_nms;
}

std::vector<FaceRect> MTCNN::BoxRegress(std::vector<FaceRect>& faceInfo, int stage)
{
    std::vector<FaceRect> bboxes;
    for (size_t bboxId = 0; bboxId < faceInfo.size(); bboxId++)
    {
    	FaceRect tempFaceInfo;
        float regh = faceInfo[bboxId].y2 - faceInfo[bboxId].y1;
        regh += (stage == 1) ? 0 : 1;
        float regw = faceInfo[bboxId].x2 - faceInfo[bboxId].x1;
        regw += (stage == 1) ? 0 : 1;
        tempFaceInfo.x1 = faceInfo[bboxId].x1 + regw * faceInfo[bboxId].regression[0];
        tempFaceInfo.y1 = faceInfo[bboxId].y1 + regh * faceInfo[bboxId].regression[1];
        tempFaceInfo.x2 = faceInfo[bboxId].x2 + regw * faceInfo[bboxId].regression[2];
        tempFaceInfo.y2 = faceInfo[bboxId].y2 + regh * faceInfo[bboxId].regression[3];
        tempFaceInfo.score = faceInfo[bboxId].score;

        tempFaceInfo.regression[0] = faceInfo[bboxId].regression[0];
        tempFaceInfo.regression[1] = faceInfo[bboxId].regression[1];
        tempFaceInfo.regression[2] = faceInfo[bboxId].regression[2];
        tempFaceInfo.regression[3] = faceInfo[bboxId].regression[3];
        if(stage == 3)
            tempFaceInfo.facepts = faceInfo[bboxId].facepts;
        bboxes.push_back(tempFaceInfo);
    }
    return bboxes;
}

void MTCNN::Bbox2Square(std::vector<FaceRect>& bboxes)
{
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        float w = bboxes[i].x2 - bboxes[i].x1;
        float h = bboxes[i].y2 - bboxes[i].y1;
        float side = h > w ? h : w;
        bboxes[i].x1 += (w-side) * 0.5;
        bboxes[i].y1 += (h-side) * 0.5;

        bboxes[i].x2 = (int)(bboxes[i].x1 + side);
        bboxes[i].y2 = (int)(bboxes[i].y1 + side);
        bboxes[i].x1 = (int)(bboxes[i].x1);
        bboxes[i].y1 = (int)(bboxes[i].y1);

    }
}

// compute the padding coordinates (pad the bounding boxes to square)
void MTCNN::Padding(int img_w, int img_h)
{
    for (size_t i = 0; i < m_regressed_rects.size(); i++)
    {
    	FaceRect tempFaceInfo;
        tempFaceInfo = m_regressed_rects[i];
        tempFaceInfo.y2 = (m_regressed_rects[i].y2 >= img_h) ? img_h : m_regressed_rects[i].y2;
        tempFaceInfo.x2 = (m_regressed_rects[i].x2 >= img_w) ? img_w : m_regressed_rects[i].x2;
        tempFaceInfo.y1 = (m_regressed_rects[i].y1 < 1) ? 1 : m_regressed_rects[i].y1;
        tempFaceInfo.x1 = (m_regressed_rects[i].x1 < 1) ? 1 : m_regressed_rects[i].x1;
        m_regressed_pading.push_back(tempFaceInfo);
    }
}


void MTCNN::ClassifyFace(const std::vector<FaceRect>& regressed_rects, cv::Mat& sample_single,
                         NetStruct* net, mx_float thresh, char netName)
{
    size_t numBox = regressed_rects.size();
    NDArray& crop_input_layer = net->InputDataNDArray();
    mx_uint input_channels = crop_input_layer.GetShape()[1];
    mx_uint input_width  = crop_input_layer.GetShape()[2];
    mx_uint input_height = crop_input_layer.GetShape()[3];

    crop_input_layer.Reshape(Shape(1, input_channels, input_width, input_height));

    m_condidate_rects.clear();

    //load crop_img data to NDArray
    for (size_t i = 0; i < numBox; i++)
    {
        size_t reg_id = 0;
        size_t confidence_id = 1;

        std::vector<cv::Mat> channels;
        NDArray input_layer(net->InputDataNDArray().GetShape(), Context::cpu());
        WrapInputLayer(&channels, &input_layer, input_width, input_height);

        int pad_top   = std::abs(m_regressed_pading[i].y1 - regressed_rects[i].y1);
        int pad_left  = std::abs(m_regressed_pading[i].x1 - regressed_rects[i].x1);
        int pad_right = std::abs(m_regressed_pading[i].x2 - regressed_rects[i].x2);
        int pad_bottom= std::abs(m_regressed_pading[i].y2 - regressed_rects[i].y2);

        cv::Mat crop_img = sample_single(cv::Range(m_regressed_pading[i].y1-1, m_regressed_pading[i].y2),
                                         cv::Range(m_regressed_pading[i].x1-1, m_regressed_pading[i].x2));

        cv::copyMakeBorder(crop_img, crop_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));

#ifdef INTER_FAST
        cv::resize(crop_img, crop_img, cv::Size(input_width,input_height), 0, 0, cv::INTER_NEAREST);
#else
        cv::resize(crop_img, crop_img, cv::Size(input_width,input_height), 0, 0, cv::INTER_AREA);
#endif

        crop_img = (crop_img-127.5)*0.0078125;

        cv::split(crop_img,channels);

        net->Forward(&input_layer);

        if(netName == 'o')
        {
            reg_id =1;
            confidence_id = 2;
        }

        const NDArray* reg = net->OutputNDArray(reg_id);
        const NDArray* confidence = net->OutputNDArray(confidence_id);
        // ONet points_offset != NULL
        const NDArray* points_offset = net->OutputNDArray(0);

        const float* confidence_data = confidence->GetData() + confidence->Size()/2;
        const float* reg_data = reg->GetData();
        const float* points_data;
        if(netName == 'o')
            points_data = points_offset->GetData();

        if (*(confidence_data) > thresh)
        {
        	FaceRect faceInfo;
            faceInfo.x1 = regressed_rects[i].x1;
            faceInfo.y1 = regressed_rects[i].y1;
            faceInfo.x2 = regressed_rects[i].x2;
            faceInfo.y2 = regressed_rects[i].y2;
            faceInfo.score = *(confidence_data);

            faceInfo.regression[0] = reg_data[0];
            faceInfo.regression[1] = reg_data[1];
            faceInfo.regression[2] = reg_data[2];
            faceInfo.regression[3] = reg_data[3];

            // x x x x x y y y y y
            if (netName == 'o')
            {
                FacePts face_pts;
                float h = faceInfo.y2 - faceInfo.y1 + 1;
                float w = faceInfo.x2 - faceInfo.x1 + 1;
                for(int j = 0; j < 5; j++)
                {
                    face_pts.y[j] = faceInfo.y1 + *(points_data + j + 5) * h - 1;
                    face_pts.x[j] = faceInfo.x1 + *(points_data + j) * w -1;
                }
                faceInfo.facepts = face_pts;
            }
            m_condidate_rects.push_back(faceInfo);
        }
    }
    m_regressed_pading.clear();
}

// multi test image pass a forward
void MTCNN::ClassifyFace_MulImage(const std::vector<FaceRect>& regressed_rects, cv::Mat& sample_single,
                                  NetStruct* net, mx_float thresh, char netName)
{
    m_condidate_rects.clear();

    size_t numBox = regressed_rects.size();
    std::vector<NDArray> data_vector;

    NDArray& input_data = net->InputDataNDArray();
    mx_uint input_channels = input_data.GetShape()[1];
    mx_uint input_width  = input_data.GetShape()[2];
    mx_uint input_height = input_data.GetShape()[3];

    NDArray mem_data(Shape(numBox, input_channels, input_width, input_height), g_GetContext());

    // load crop_img data to NDArray
    for (size_t i = 0; i < numBox; i++)
    {
        int pad_top   = std::abs(m_regressed_pading[i].x1 - regressed_rects[i].x1);
        int pad_left  = std::abs(m_regressed_pading[i].y1 - regressed_rects[i].y1);
        int pad_right = std::abs(m_regressed_pading[i].y2 - regressed_rects[i].y2);
        int pad_bottom= std::abs(m_regressed_pading[i].x2 - regressed_rects[i].x2);

        cv::Mat crop_img = sample_single(cv::Range(m_regressed_pading[i].y1-1, m_regressed_pading[i].y2),
                                         cv::Range(m_regressed_pading[i].x1-1, m_regressed_pading[i].x2));
        cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));

#ifdef INTER_FAST
        cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_NEAREST);
#else
        cv::resize(crop_img, crop_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_AREA);
#endif

        crop_img = (crop_img-127.5)*0.0078125;
        NDArray data(Shape(1, input_channels, input_width, input_height), Context::cpu());
        CvMatToNDArraySignalChannel(crop_img, &data);
        data_vector.push_back(data);
    }
    m_regressed_pading.clear();

    /* extract the features and store */
    AddNDArrayVector(&mem_data, data_vector);
    /* fire the network */
    net->Forward(&mem_data);

    size_t reg_id = 0;
    size_t confidence_id = 1;
    if (netName == 'o')
    {
        reg_id =1;
        confidence_id = 2;
    }

    const NDArray* reg = net->OutputNDArray(reg_id);
    const NDArray* confidence = net->OutputNDArray(confidence_id);
    // ONet points_offset != NULL
    const NDArray* points_offset = net->OutputNDArray(0);

    const float* confidence_data = confidence->GetData();
    const float* reg_data = reg->GetData();
    const float* points_data;
    if (netName == 'o')
        points_data = points_offset->GetData();


    for (size_t i = 0; i < numBox; i++)
    {
        if (*(confidence_data+i*2+1) > thresh)
        {
        	FaceRect faceInfo;
            faceInfo.x1 = regressed_rects[i].x1;
            faceInfo.y1 = regressed_rects[i].y1;
            faceInfo.x2 = regressed_rects[i].x2;
            faceInfo.y2 = regressed_rects[i].y2;
            faceInfo.score  = *(confidence_data+i*2+1);

            faceInfo.regression[0] = reg_data[4*i+0];
            faceInfo.regression[1] = reg_data[4*i+1];
            faceInfo.regression[2] = reg_data[4*i+2];
            faceInfo.regression[3] = reg_data[4*i+3];

            if(netName == 'o')
            {
                FacePts face_pts;
                float h = faceInfo.y2 - faceInfo.y1 + 1;
                float w = faceInfo.x2 - faceInfo.x1 + 1;
                for(int j = 0; j < 5; j++)
                {
                    face_pts.y[j] = faceInfo.y1 + *(points_data+j+5+10*i) * h - 1;
                    face_pts.x[j] = faceInfo.x1 + *(points_data+j+10*i) * w -1;
                }
                faceInfo.facepts = face_pts;
            }
            m_condidate_rects.push_back(faceInfo);
        }
    }
}
bool MTCNN::CvMatToNDArraySignalChannel(const cv::Mat& cv_mat, NDArray* data)
{
    if (cv_mat.empty())
        return false;

    int size = cv_mat.rows * cv_mat.cols * cv_mat.channels();
    mx_float* image_data = const_cast<mx_float*>(data->GetData());
    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    for (int i = 0; i < cv_mat.rows; i++)
    {
        const float * data = cv_mat.ptr<float>(i);
        for (int j = 0; j < cv_mat.cols; j++)
        {
            *ptr_image_b++ = *data++;
            *ptr_image_g++ = *data++;
            *ptr_image_r++ = *data++;
        }
    }
    return true;
}

void MTCNN::AddNDArrayVector(NDArray* data, std::vector<NDArray> data_vector)
{
    size_t w = data->GetShape()[2];
    size_t h = data->GetShape()[3];
    size_t ch = data->GetShape()[1];
    size_t num = data->GetShape()[0];
    size_t size = data->Size()/num;
    NDArray tmp(Shape(num, ch, w, h), Context::cpu());
    for (size_t i = 0; i < data_vector.size(); ++i)
    {
        mx_float* src = const_cast<mx_float*>(tmp.GetData());
        memcpy(&src[i*size], data_vector[i].GetData(), size*sizeof(float));
    }
    tmp.CopyTo(data);
}
