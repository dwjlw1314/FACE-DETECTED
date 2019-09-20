#include "gsfacedetectsdk.h"
#include "mtcnn_net.h"
#include "RetinaFace.h"
#include "featurextract.h"

//2.0.0.0 use #define value
#define VERSION "2.0.0.0"

#define USED_WARPAFFINEI 1
#define USED_BGRToRGB 0
#define USED_RETINAFACE 1
#define USED_MTCNN 0
#define USED_TVM 0
using namespace cv;

int g_DeviceId;
DeviceType g_DeviceType;

void g_SetContext(int id = 0, DeviceType type = mxnet::cpp::kCPU)
{
	g_DeviceId = id;
    g_DeviceType = type;
}

Context g_GetContext()
{
    if (g_DeviceType == mxnet::cpp::kGPU)
    {
        return Context::gpu(g_DeviceId);
    }
    return Context::cpu(g_DeviceId);
}

GsFaceDetectSDK::GsFaceDetectSDK(const char* path, MtcnnPar *param, bool useGpu)
	: m_mtcnn(nullptr), m_retinaFace(nullptr), m_feaExt(nullptr)
{
	//init default par member
	MtcnnPar default_par {0, 40, 0.509,
					{112,112},
					{0.6,0.8,0.9},
					{0.5,0.7,0.7},
					"fc1_output"};
	if (!param)
		param = &default_par;

	if (useGpu)
		g_SetContext(param->devid, mxnet::cpp::kGPU);
	else
		g_SetContext(param->devid, mxnet::cpp::kCPU);

	m_OutWidth = param->featureshape[0];
	m_OutHeight = param->featureshape[1];
#if USED_RETINAFACE
	m_retinaFace = new RetinaFace(path, param, useGpu);
#elif USED_MTCNN
    m_mtcnn = new MTCNN(path, param, useGpu);
#endif
    m_feaExt = new FeaturExtract(path, param);
}

void GsFaceDetectSDK::warpAffineI(cv::Mat& image, FaceRect& faceRect, std::vector<cv::Mat>& featureMat)
{
	float landmarks[5][2] = {{30.2946f + 8.0, 51.6963f},
							{65.5318f + 8.0, 51.5014f},
							{48.0252f + 8.0, 71.7366f},
							{33.5493f + 8.0, 92.3655f},
							{62.7299f + 8.0, 92.2041f}};
	//对齐点的Point2f数组,检测到的人脸对齐点，注意这里是基于原始图像的坐标点
	cv::Point2f srcTri[5];
	//对齐点的Point2f数组,模板的Landmarks，注意这是一个基于输出图像大小尺寸的坐标点
	cv::Point2f destTri[5];

	for (int i = 0; i < 5; i++)
	{
		srcTri[i] = Point2f(faceRect.facepts.x[i]-faceRect.x1, faceRect.facepts.y[i]-faceRect.y1);
		destTri[i] = Point2f(landmarks[i][0] , landmarks[i][1]);
	}

	Mat warp_frame(m_OutWidth, m_OutHeight, CV_32FC3);

	Mat warp_mat = getAffineTransform(srcTri, destTri);
	//使用相似变换，不适合使用仿射变换，会导致图像变形,ref:opencv_videoio
	//Mat warp_ma = estimateRigidTransform(image, warp_frame, false);

	warpAffine(image(cv::Range(faceRect.y1, faceRect.y2), cv::Range(faceRect.x1, faceRect.x2)),
			warp_frame, warp_mat, warp_frame.size(), 1, 0, 0);

	featureMat.push_back(warp_frame);
}

#if USED_BGRToRGB
/*
 * picture channel = 3
 * Mat BGR转RGB
 */
cv::Mat GsFaceDetectSDK::BGRToRGB(cv::Mat& img)
{
	cv::Mat image(img.rows, img.cols, CV_8UC3);
	for(int i = 0; i < img.rows; ++i)
	{
		//获取第i行首像素指针
		cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
		cv::Vec3b *p2 = image.ptr<cv::Vec3b>(i);
		for(int j = 0; j < img.cols; ++j)
		{
			//将img的bgr转为image的rgb
			p2[j][2] = p1[j][0];
			p2[j][1] = p1[j][1];
			p2[j][0] = p1[j][2];
		}
	}
	return image;
}
#endif

/*
 * 通过最后一个参数指定类型，获取不同的数据
 */
FACE_NUMBERS GsFaceDetectSDK::getFacesAllResult(const uchar* data, int width, int height,
		int channel, pFaceRect facePoints, GetFaceType type)
{
    FACE_NUMBERS ret = 0;
    std::vector<FaceRect> faceInfo;
    std::vector<cv::Mat> FeatureMat;
    cv::Mat image(height, width, CV_8UC3);

    if(nullptr == data)
    	return ret;
    //Mat img_rgb;
    for(int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image.at<cv::Vec3b>(i,j)[0] = data[width * channel * i + j * channel + 0];
            image.at<cv::Vec3b>(i,j)[1] = data[width * channel * i + j * channel + 1];
            image.at<cv::Vec3b>(i,j)[2] = data[width * channel * i + j * channel + 2];
        }
    }

#if USED_BGRToRGB
	image = BGRToRGB(image);
#endif

    if (GetFaceType::FACEBOX & type)
    {
#if USED_RETINAFACE
    	m_retinaFace->Detect(image, faceInfo);
#elif USED_MTCNN
		m_mtcnn->Detect(image, faceInfo);
#endif
		for(auto face : faceInfo)
		{
			facePoints[ret++] = face;
		}
    }

	if(0 == ret) return ret;

	if (GetFaceType::FACEFEATURE & type)
	{
		/*
		 * 截取原图的指定位置大小的区域  dst_img = src_img(Range(0,100),Range(50,200));
		 * 这里截取的就是原图第0行至第99行,第50列至199列的区域图像.这里要注意的就是Range的两个参数范围分别为左包含和右不包含
		*/
		for (size_t i = 0; i < ret; i++)
		{
			facePoints[i].y1 = facePoints[i].y1 <= 0 ? 0 : facePoints[i].y1;
			facePoints[i].x1 = facePoints[i].x1 <= 0 ? 0 : facePoints[i].x1;
			facePoints[i].y2 = facePoints[i].y2 >= height ? height : facePoints[i].y2;
			facePoints[i].x2 = facePoints[i].x2 >= width ? width : facePoints[i].x2;

#if USED_WARPAFFINEI
			warpAffineI(image, facePoints[i], FeatureMat);
#else
			int x1 = facePoints[i].x1;
			int x2 = facePoints[i].x2;
			int y1 = facePoints[i].y1;
			int y2 = facePoints[i].y2;
			FeatureMat.push_back(image(cv::Range(y1, y2), cv::Range(x1, x2)));
#endif
		}

		m_feaExt->GetFaceFeature(FeatureMat, facePoints);
    }

    return ret;
}

FACE_NUMBERS GsFaceDetectSDK::getFacesAllResult(cv::Mat image, pFaceRect facePoints, GetFaceType type)
{
	FACE_NUMBERS ret = 0;
	int width = image.cols;
	int height = image.rows;
	std::vector<cv::Mat> FeatureMat;
	std::vector<FaceRect> faceInfo;

    if(image.cols < 30 || image.rows < 30)
    	return ret;

#if USED_BGRToRGB
	image = BGRToRGB(image);
#endif

	if (GetFaceType::FACEBOX & type)
	{
#if USED_RETINAFACE
    	m_retinaFace->Detect(image, faceInfo);
#elif USED_MTCNN
		m_mtcnn->Detect(image, faceInfo);
#endif
		for(auto face : faceInfo)
		{
			facePoints[ret++] = face;
		}
	}

	if(0 == ret) return ret;

	if (GetFaceType::FACEFEATURE & type)
	{
		/*
		 * 截取原图的指定位置大小的区域  dst_img = src_img(Range(0,100),Range(50,200));
		 * 这里截取的就是原图第0行至第99行,第50列至199列的区域图像.这里要注意的就是Range的两个参数范围分别为左包含和右不包含
		*/
		for (size_t i = 0; i < ret; i++)
		{
			facePoints[i].y1 = facePoints[i].y1 <= 0 ? 0 : facePoints[i].y1;
			facePoints[i].x1 = facePoints[i].x1 <= 0 ? 0 : facePoints[i].x1;
			facePoints[i].y2 = facePoints[i].y2 > height ? height : facePoints[i].y2;
			facePoints[i].x2 = facePoints[i].x2 > width ? width : facePoints[i].x2;

#if USED_WARPAFFINEI
			warpAffineI(image, facePoints[i], FeatureMat);
#else
			int x1 = facePoints[i].x1;
			int x2 = facePoints[i].x2;
			int y1 = facePoints[i].y1;
			int y2 = facePoints[i].y2;
			FeatureMat.push_back(image(cv::Range(y1, y2), cv::Range(x1, x2)));
#endif
		}

		m_feaExt->GetFaceFeature(FeatureMat, facePoints);
	}
	return ret;
}

bool GsFaceDetectSDK::getFacesFeatureResult(cv::Mat frame, float* feature)
{
	std::vector<cv::Mat> FeatureMat;
	FaceRect face_point;

#if USED_BGRToRGB
	frame = BGRToRGB(frame);
#endif

	/*
	 * 截取原图的指定位置大小的区域  dst_img = src_img(Range(0,100),Range(50,200));
	 * 这里截取的就是原图第0行至第99行,第50列至199列的区域图像.这里要注意的就是Range的两个参数范围分别为左包含和右不包含
	*/
	FeatureMat.push_back(frame);
	m_feaExt->GetFaceFeature(FeatureMat, &face_point);

	memset(feature,0,FSIZE);

	for(size_t i = 0; i < FSIZE; i++)
		feature[i] = face_point.feature[i];

	return true;
}

void GsFaceDetectSDK::ReleaseResource(void)
{
#if USED_RETINAFACE
	delete m_retinaFace;
#elif USED_MTCNN
	delete m_mtcnn;
#endif
	delete m_feaExt;
	//MXNotifyShutdown();
}

int GsFaceDetectSDK::getGPUCount(int* out)
{
	return MXGetGPUCount(out);
}

const char* GsFaceDetectSDK::getSDKVersion()
{
	string version = VERSION;
	return version.c_str();
}

float GsFaceDetectSDK::compares(const float* feature1, const float* feature2)
{
    return m_feaExt->getSimilarity(feature1, feature2);
}
