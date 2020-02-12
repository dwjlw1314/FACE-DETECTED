#include "gsfacedetectsdk.h"
#include "mtcnn_net.h"
#include "RetinaFace.h"
#include "featurextract.h"
#include "AgeSexExtract.h"

//2.0.1.4 use #define value
#define VERSION "2.0.1.4"

#define USED_TRACKER 0
#define USED_WARPAFFINEI 1
#define USED_BGRToRGB 0
#define USED_RETINAFACE 1
#define USED_MTCNN 0
#define USED_TVM 0
using namespace cv;

int g_DeviceId;
DeviceType g_DeviceType;

bool g_EnableRelease = false;
//sync_mutex
/*
 * set PTHREAD_MUTEX_INITIALIZER running error
 * __pthread_tpp_change_priority: Assertion 'new_prot == -1 ||
 * (new_prot >= fifo_min_prio && new_prot <= fifo_max_prio') failed
 */
pthread_mutex_t g_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

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

MtcnnPar default_par{0, 40, 0.509,	{56,56}, {112,112}, {0.6,0.8,0.9},
	{0.4,0.7,0.7}, 1920, FACTOR_16_9, (char*)"fc1_output", FACEALL};

float landmark_56[5][2] = {{30.2946f/2 + 4.0, 51.6963f/2},
						{65.5318f/2 + 4.0, 51.5014f/2},
						{48.0252f/2 + 4.0, 71.7366f/2},
						{33.5493f/2 + 4.0, 92.3655f/2},
						{62.7299f/2 + 4.0, 92.2041f/2}};

float landmark_112[5][2] = {{30.2946f + 8.0, 51.6963f},
						{65.5318f + 8.0, 51.5014f},
						{48.0252f + 8.0, 71.7366f},
						{33.5493f + 8.0, 92.3655f},
						{62.7299f + 8.0, 92.2041f}};

GsFaceDetectSDK::GsFaceDetectSDK(const char* path, MtcnnPar *param, bool useGpu)
	: m_mtcnn(nullptr), m_retinaFace(nullptr), m_feaExt(nullptr), m_genderExt(nullptr)
{
	//init default par member
	if (!param)
		param = &default_par;

	m_pictureWSize = param->LongSideSize;
	switch(param->scalRatio)
	{
		case FACTOR_8_16:
			m_pictureHSize = m_pictureWSize*2; break;
		case FACTOR_4_3:
			m_pictureHSize = m_pictureWSize*3/4; break;
		case FACTOR_16_9:
			m_pictureHSize = m_pictureWSize*9/16; break;
		case FACTOR_16_10:
			m_pictureHSize = m_pictureWSize*10/16; break;
	}

	m_fshape[0] = param->featureshape[0];
	m_fshape[1] = param->featureshape[1];
	m_gshape[0] = param->gendershape[0];
	m_gshape[1] = param->gendershape[1];

	pthread_mutex_lock(&g_mutex);

	try {
		if(param->type & FACEBOX)
		{
#if USED_RETINAFACE
			m_retinaFace = new RetinaFace(path, param, useGpu);
#elif USED_MTCNN
			m_mtcnn = new MTCNN(path, param, useGpu);
#endif
		}
		if(param->type & FACEFEATURE)
			m_feaExt = new FeaturExtract(path, param, useGpu);
		if(param->type & FACEGENDER)
			m_genderExt = new AgeSexExtract(path, param, useGpu);
		m_InitStatus = true;
	}
	catch(/*dmlc::Error &err*/...)
	{
		//cout << "dmlc::Error: " << err.what() << endl;
		m_InitStatus = false;
	}

    //前期调试使用，后期发布可以使用自解锁模式
	pthread_mutex_unlock(&g_mutex);
}

void GsFaceDetectSDK::warpAffineI(cv::Mat& image, FaceRect& faceRect, std::vector<cv::Mat>& featureMat, GetFaceType type)
{
	size_t shape[2];
	float (*landmarks)[5][2];

	if (type == FACEGENDER)
	{
		shape[0] = m_gshape[0];
		shape[1] = m_gshape[1];
	}
	else
	{
		shape[0] = m_fshape[0];
		shape[1] = m_fshape[1];
	}

	if (shape[0] == 56)
		landmarks = &landmark_56;
	else
		landmarks = &landmark_112;

	//对齐点的Point2f数组,检测到的人脸对齐点，注意这里是基于原始图像的坐标点
	MatrixXd srcM(5,2);
	//对齐点的Point2f数组,模板的Landmarks，注意这是一个基于输出图像大小尺寸的坐标点
	MatrixXd dstM(5,2);

	for(int i = 0; i < 5; i++)
	{
		srcM(i,0) = faceRect.facepts.x[i]-faceRect.x1;
		srcM(i,1) = faceRect.facepts.y[i]-faceRect.y1;
		dstM(i,0) = (*landmarks)[i][0];
		dstM(i,1) = (*landmarks)[i][1];
	}

	Mat warp_frame(shape[0], shape[1], image.type());

	Mat warp_mat = similizerTransform(srcM, dstM, true).clone();

	warpAffine(image(cv::Range(faceRect.y1, faceRect.y2), cv::Range(faceRect.x1, faceRect.x2)),
			warp_frame, warp_mat, warp_frame.size(), 2, 0, 0);

	featureMat.push_back(warp_frame);
}

void GsFaceDetectSDK::warpAffineI(cv::Mat& image, FacePts& facepts, std::vector<cv::Mat>& featureMat, GetFaceType type)
{
	size_t shape[2];
	float (*landmarks)[5][2];

	if (type == FACEGENDER)
	{
		shape[0] = m_gshape[0];
		shape[1] = m_gshape[1];
	}
	else
	{
		shape[0] = m_fshape[0];
		shape[1] = m_fshape[1];
	}

	if (shape[0] == 56)
		landmarks = &landmark_56;
	else
		landmarks = &landmark_112;

	//对齐点的Point2f数组,检测到的人脸对齐点，注意这里是基于原始图像的坐标点
	MatrixXd srcM(5,2);
	//对齐点的Point2f数组,模板的Landmarks，注意这是一个基于输出图像大小尺寸的坐标点
	MatrixXd dstM(5,2);

	for(int i = 0; i < 5; i++)
	{
		srcM(i,0) = facepts.x[i];
		srcM(i,1) = facepts.y[i];
		dstM(i,0) = (*landmarks)[i][0];
		dstM(i,1) = (*landmarks)[i][1];
	}

	Mat warp_frame(shape[0], shape[1], image.type());

    Mat warp_mat = similizerTransform(srcM, dstM, true).clone();

	warpAffine(image, warp_frame, warp_mat, warp_frame.size(), 2, 0, 0);

	featureMat.push_back(warp_frame);
}

Mat GsFaceDetectSDK::similizerTransform(MatrixXd& src, MatrixXd& dst, bool estimate_scale)  //输入：目标点，原图点，是否scale（这里为true）
{
	int num = 5;
	int dim = 2;

	MatrixXd src_mean(5,2);
	MatrixXd dst_mean(5,2);

	src_mean(0,0) = src.col(0).mean();
	src_mean(0,1) = src.col(1).mean();
	dst_mean(0,0) = dst.col(0).mean();
	dst_mean(0,1) = dst.col(1).mean();
	src_mean.col(0).fill(src_mean(0,0));
	src_mean.col(1).fill(src_mean(0,1));
	dst_mean.col(0).fill(dst_mean(0,0));
	dst_mean.col(1).fill(dst_mean(0,1));

	MatrixXd src_demean(5,2);
	MatrixXd dst_demean(5,2);

	src_demean = src.array() - src_mean.array();
	dst_demean = dst.array() - dst_mean.array();

	MatrixXd A;
	A = (dst_demean.transpose() * src_demean) / num;

	MatrixXd d;
	d.setOnes(1, dim);

	if (A.determinant() < 0)
		d(0, dim - 1) = -1;

	MatrixXd T;
	T.setIdentity(3,3);

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
	Eigen::MatrixXd S = svd.singularValues();
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();

	FullPivLU<MatrixXd> lu(A);
	auto rank = lu.rank();

	if (rank == 0)
	{
		//如果A的秩为0， 返回T的无效值
		Mat mat;
		return mat;
	}
	else if (rank == dim - 1)
	{
		if (U.determinant() * V.determinant() > 0) //#矩阵U、V的行列式的值
		{
			auto tmp = U * V;
			T.topLeftCorner(dim,dim) << tmp(0,0),tmp(0,1),tmp(1,0),tmp(1,1);  //# @为矩阵点乘
		}
		else
		{
			auto s = d(0, dim - 1);
			d(0, dim - 1) = -1;
			//np.diag是将矩阵d[1., 1.]改写为[1., 0; 0, 1.]
			MatrixXd tmp_;
			tmp_.setZero(dim,dim);
			tmp_(0,0) = d(0,0);
			tmp_(1,1) = d(0,1);
			auto tmp = U * tmp_ * V;
			T.topLeftCorner(dim,dim) << tmp(0,0),tmp(0,1),tmp(1,0),tmp(1,1);
			d(0, dim - 1) = s;
		}
	}
	else
	{
		MatrixXd tmp_;
		tmp_.setZero(dim,dim);
		tmp_(0,0) = d(0,0);
		tmp_(1,1) = d(0,1);
		auto tmp = U * tmp_ * V;
		T.topLeftCorner(dim,dim) << tmp(0,0),tmp(0,1),tmp(1,0),tmp(1,1);;
	}
	float scale;
    if (estimate_scale)
    {
       //# .var(axis=0).sum() 是求src_demean矩阵在列方向上的方差，后再求和
    	Eigen::MatrixXd mean = src_demean.colwise().mean();

    	Eigen::MatrixXd sqsum_ = src_demean.transpose() * src_demean;

    	Eigen::MatrixXd ide = MatrixXd::Identity(2,2);
//    	Eigen::MatrixXd temp = sqsum_.cwiseProduct(ide);
//    	std::cout << sqsum_.array() * ide.array() << std::endl;
    	Eigen::MatrixXd sqsum = sqsum_.cwiseProduct(ide).colwise().sum();

    	Eigen::MatrixXd scale_(1,2);
    	float _scale_ = 1. / num;
    	scale_ <<  _scale_, _scale_;
    	Eigen::MatrixXd variance_ = sqsum .array()* scale_.array() - mean.array() * mean.array();

    	float _variance_ = variance_.sum();
    	_variance_ = 1. / _variance_;

    	scale = _variance_ * (d * S)(0,0);
    }
    else
    {
    	scale = 1.0;
    }

    Eigen::MatrixXd ss(2,1);
    ss << scale,scale;
    Eigen::MatrixXd src_mean_ = (ss.array() * (T.topLeftCorner(dim,dim) * src_mean.topLeftCorner(1,2).transpose()).array());
    Eigen::MatrixXd dst_mean_ = dst_mean.topLeftCorner(1,2).transpose();

    Eigen::MatrixXd T_ = dst_mean_.array() - src_mean_.array();
    T(0,2) = T_(0,0);
    T(1,2) = T_(1,0);

    Eigen::MatrixXd sss(2,2);
    sss << scale,scale,scale,scale;
    Eigen::MatrixXd _T__ = T.topLeftCorner(2,2);
    Eigen::MatrixXd _T_ = _T__.cwiseProduct(sss);
    T.topLeftCorner(2,2) << _T_(0,0),_T_(0,1),_T_(1,0),_T_(1,1);

    Eigen::MatrixXd M = T.topLeftCorner(2,3);

	//#最终结果，我们要的矩阵M 3*3 to 2*3
	cv::Mat rr;
	cv::eigen2cv(M,rr);

    return rr;
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

float GsFaceDetectSDK::normalization(cv::Mat& src)
{
	Mat temImage;
    Mat imageROI;
    float scalfactor = 1.;

	int w = src.cols;
	int h = src.rows;

	Scalar color = Scalar(0,0,0);

	if (w > m_pictureWSize || h > m_pictureHSize)
	{
		int wf = w-m_pictureWSize;
		int hf = h-m_pictureHSize;
		if (wf > hf)
			scalfactor = float(m_pictureWSize)/(float)w;
		else
			scalfactor = float(m_pictureHSize)/(float)h;
		resize(src,src,Size(w*scalfactor, h*scalfactor), 0, 0, cv::INTER_CUBIC);
	}
    copyMakeBorder(src, src, 0, m_pictureHSize-src.rows, 0, m_pictureWSize-src.cols, BORDER_CONSTANT, color);

    /* 网络摄像头使用
	int kernel_size = 3; //滤波器的核
	Mat kern = Mat::ones(kernel_size,kernel_size,CV_32F)/(float)(kernel_size*kernel_size);

	//由于原视频是网络摄像头采集的，所以有很多雪花点，在这里进行了简单的均值滤波处理
	filter2D(src,src,-1,kern);
    */

    return scalfactor;
}

/*
 * 通过最后一个参数指定类型，获取不同的数据
 */
FACE_NUMBERS GsFaceDetectSDK::getFacesAllResult(const uchar* data, int width, int height,
		int channel, pFaceRect facePoints, bool dynamic_scale, GetFaceType type)
{
	while (g_EnableRelease)
		usleep(1000);

	float scalfactor = 1.;
    FACE_NUMBERS ret = 0;
    std::vector<FaceRect> faceInfo;
    std::vector<cv::Mat> FeatureMat;
    std::vector<cv::Mat> GenderMat;
    cv::Mat image(height, width, CV_8UC3);

    if(!m_InitStatus || nullptr == data || nullptr == facePoints)
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

	if ((GetFaceType::FACEBOX & type) && (m_retinaFace || m_mtcnn))
    {
    	if (dynamic_scale)
    	{
    		scalfactor = normalization(image);
    	}
    	try {
#if USED_RETINAFACE
    		//vector<FaceRect>().swap(faceInfo);
    		m_retinaFace->Detect(image, faceInfo);
#elif USED_MTCNN
    		m_mtcnn->Detect(image, faceInfo);
#endif
		}
		catch(/*dmlc::Error &err*/...)
		{
			ret = 0;
			faceInfo.clear();
			//是否重新抛出dmlc::Error异常到上层函数,根据业务需求定
		}
		for(auto face : faceInfo)
		{
			facePoints[ret++] = face;
		}

		if(0 == ret) return ret;
    }

    if ((GetFaceType::FACEFEATURE & type) && m_feaExt)
	{
		size_t i = 0;
		if(0 == ret)
		{
			while(facePoints[i++].score > 0.6)
				ret++;
		}
		/*
		 * 截取原图的指定位置大小的区域  dst_img = src_img(Range(0,100),Range(50,200));
		 * 这里截取的就是原图第0行至第99行,第50列至199列的区域图像.这里要注意的就是Range的两个参数范围分别为左包含和右不包含
		*/
		for (size_t i = 0; i < ret; i++)
		{
			facePoints[i].y1 = facePoints[i].y1 <= 0 ? 0 : facePoints[i].y1;
			facePoints[i].x1 = facePoints[i].x1 <= 0 ? 0 : facePoints[i].x1;
			facePoints[i].y2 = facePoints[i].y2 >= image.rows ? image.rows : facePoints[i].y2;
			facePoints[i].x2 = facePoints[i].x2 >= image.cols ? image.cols : facePoints[i].x2;

#if USED_WARPAFFINEI
			warpAffineI(image, facePoints[i], FeatureMat, FACEFEATURE);
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

    if ((GetFaceType::FACEGENDER & type) && m_genderExt)
    {
		size_t i = 0;
		if(0 == ret)
		{
			while(facePoints[i++].score > 0.6)
				ret++;
		}
		/*
		 * 截取原图的指定位置大小的区域  dst_img = src_img(Range(0,100),Range(50,200));
		 * 这里截取的就是原图第0行至第99行,第50列至199列的区域图像.这里要注意的就是Range的两个参数范围分别为左包含和右不包含
		*/
		for (size_t i = 0; i < ret; i++)
		{
			facePoints[i].y1 = facePoints[i].y1 <= 0 ? 0 : facePoints[i].y1;
			facePoints[i].x1 = facePoints[i].x1 <= 0 ? 0 : facePoints[i].x1;
			facePoints[i].y2 = facePoints[i].y2 >= image.rows ? image.rows : facePoints[i].y2;
			facePoints[i].x2 = facePoints[i].x2 >= image.cols ? image.cols : facePoints[i].x2;

#ifdef USED_WARPAFFINEI
			warpAffineI(image, facePoints[i], GenderMat, FACEGENDER);
#else
			int x1 = facePoints[i].x1;
			int x2 = facePoints[i].x2;
			int y1 = facePoints[i].y1;
			int y2 = facePoints[i].y2;
			GenderMat.push_back(image(cv::Range(y1, y2), cv::Range(x1, x2)));
#endif
		}
		m_genderExt->GetFaceAgeSex(GenderMat, facePoints);
    }

	if (dynamic_scale)
	{
		for (size_t i = 0; i < ret; i++)
		{
			facePoints[i].y1 = facePoints[i].y1 / scalfactor;
			facePoints[i].x1 = facePoints[i].x1 / scalfactor;
			facePoints[i].y2 = facePoints[i].y2 / scalfactor;
			facePoints[i].x2 = facePoints[i].x2 / scalfactor;
			for (int j = 0; j < 5; j++)
			{
				facePoints[i].facepts.x[j] = facePoints[i].facepts.x[j] / scalfactor;
				facePoints[i].facepts.y[j] = facePoints[i].facepts.y[j] / scalfactor;
			}
		}
	}

    return ret;
}

FACE_NUMBERS GsFaceDetectSDK::getFacesAllResult(cv::Mat image, pFaceRect facePoints, bool dynamic_scale, GetFaceType type)
{
	while (g_EnableRelease)
		usleep(1000);

	float scalfactor = 1.;
	FACE_NUMBERS ret = 0;
	int width = image.cols;
	int height = image.rows;
	std::vector<cv::Mat> FeatureMat;
	std::vector<cv::Mat> GenderMat;
	std::vector<FaceRect> faceInfo;

	if (!m_InitStatus || image.empty() || nullptr == facePoints)
		return ret;

    if(image.cols < 30 || image.rows < 30)
    	return ret;

#if USED_BGRToRGB
	image = BGRToRGB(image);
#endif

	if ((GetFaceType::FACEBOX & type) && (m_retinaFace || m_mtcnn))
	{
		if (dynamic_scale)
		{
			scalfactor = normalization(image);
		}
		try {
#if USED_RETINAFACE
			m_retinaFace->Detect(image, faceInfo);
#elif USED_MTCNN
			m_mtcnn->Detect(image, faceInfo);
#endif
		}
		catch(/*dmlc::Error &err*/...)
		{
			ret = 0;
			faceInfo.clear();
		}
		for(auto face : faceInfo)
		{
			facePoints[ret++] = face;
		}

		if(0 == ret) return ret;
	}

	if ((GetFaceType::FACEFEATURE & type) && m_feaExt)
	{
		if(0 == ret)
		{
			size_t i = 0;
			while(facePoints[i++].score > 0.6)
				ret++;
		}
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
			warpAffineI(image, facePoints[i], FeatureMat, FACEFEATURE);
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

    if ((GetFaceType::FACEGENDER & type) && m_genderExt)
    {
		size_t i = 0;
		if(0 == ret)
		{
			while(facePoints[i++].score > 0.6)
				ret++;
		}
		/*
		 * 截取原图的指定位置大小的区域  dst_img = src_img(Range(0,100),Range(50,200));
		 * 这里截取的就是原图第0行至第99行,第50列至199列的区域图像.这里要注意的就是Range的两个参数范围分别为左包含和右不包含
		*/
		for (size_t i = 0; i < ret; i++)
		{
			facePoints[i].y1 = facePoints[i].y1 <= 0 ? 0 : facePoints[i].y1;
			facePoints[i].x1 = facePoints[i].x1 <= 0 ? 0 : facePoints[i].x1;
			facePoints[i].y2 = facePoints[i].y2 >= image.rows ? image.rows : facePoints[i].y2;
			facePoints[i].x2 = facePoints[i].x2 >= image.cols ? image.cols : facePoints[i].x2;

#ifdef USED_WARPAFFINEI
			warpAffineI(image, facePoints[i], GenderMat, FACEGENDER);
#else
			int x1 = facePoints[i].x1;
			int x2 = facePoints[i].x2;
			int y1 = facePoints[i].y1;
			int y2 = facePoints[i].y2;
			GenderMat.push_back(image(cv::Range(y1, y2), cv::Range(x1, x2)));
#endif
		}
		m_genderExt->GetFaceAgeSex(GenderMat, facePoints);
    }

	if (dynamic_scale)
	{
		for (size_t i = 0; i < ret; i++)
		{
			facePoints[i].y1 = facePoints[i].y1 / scalfactor;
			facePoints[i].x1 = facePoints[i].x1 / scalfactor;
			facePoints[i].y2 = facePoints[i].y2 / scalfactor;
			facePoints[i].x2 = facePoints[i].x2 / scalfactor;
			for (int j = 0; j < 5; j++)
			{
				facePoints[i].facepts.x[j] = facePoints[i].facepts.x[j] / scalfactor;
				facePoints[i].facepts.y[j] = facePoints[i].facepts.y[j] / scalfactor;
			}
		}
	}

	return ret;
}

bool GsFaceDetectSDK::getFacesFeatureResult(cv::Mat& frame, FacePts& facepts, float* feature)
{
	std::vector<cv::Mat> FeatureMat;
	FaceRect face_point;

	if (!m_InitStatus || frame.empty() || nullptr == m_feaExt)
		return false;

#if USED_BGRToRGB
	frame = BGRToRGB(frame);
#endif

#if USED_WARPAFFINEI
	warpAffineI(frame, facepts, FeatureMat, FACEFEATURE);
#else
	FeatureMat.push_back(frame);
#endif

	m_feaExt->GetFaceFeature(FeatureMat, &face_point);

//	if (abs(face_point.score) < 0.4)
//		return false;

	memset(feature,0,FSIZE);

	for(size_t i = 0; i < FSIZE; i++)
		feature[i] = face_point.feature[i];

	return true;
}

bool GsFaceDetectSDK::getFacesAgeSexResult(cv::Mat& frame, FacePts& facepts, FaceRect& facerect)
{
	std::vector<cv::Mat> GenderMat;

	if (!m_InitStatus || frame.empty() || nullptr == m_genderExt)
		return false;

#if USED_BGRToRGB
	frame = BGRToRGB(frame);
#endif

#ifdef USED_WARPAFFINEI
	warpAffineI(frame, facepts, GenderMat, FACEGENDER);
#else
	GenderMat.push_back(frame);
#endif

	m_genderExt->GetFaceAgeSex(GenderMat, &facerect);

	return true;
}

void GsFaceDetectSDK::ReleaseResource(int id)
{
#if USED_RETINAFACE
	if (m_retinaFace)
		delete m_retinaFace;
#elif USED_MTCNN
	if (m_mtcnn)
		delete m_mtcnn;
#endif
	if (m_feaExt)
		delete m_feaExt;
	if (m_genderExt)
		delete m_genderExt;
	pthread_mutex_lock(&g_mutex);
	g_EnableRelease = true;
	usleep(100000);
	MXStorageEmptyCache(mxnet::cpp::kGPU, id);
	g_EnableRelease = false;
	pthread_mutex_unlock(&g_mutex);
}

int GsFaceDetectSDK::getGPUCount(int* out)
{
	return MXGetGPUCount(out);
}

const char* GsFaceDetectSDK::getSDKVersion(bool select)
{
	string version = VERSION;
	if (!select)
	{
		int ver = 0;
		MXGetVersion(&ver);
		version = std::to_string(ver);
	}
	return version.c_str();
}

float GsFaceDetectSDK::compares(const float* feature1, const float* feature2, int fnum)
{
	float sim = 0.0;
	if (nullptr == m_feaExt)
		return sim;

//	if (1 == fnum)
//	{
//		sim = m_feaExt->getSimilarity(feature1, feature2);
//		return sim;
//	}

	for(int i = 0; i < fnum; i++)
	{
		float sim_ = m_feaExt->getSimilarity(feature1, feature2);
		sim = sim_ >  sim ? sim_ : sim;
		feature2 += FSIZE;
	}

	return sim;
}
