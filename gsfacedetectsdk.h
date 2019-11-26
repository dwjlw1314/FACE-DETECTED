#ifndef GSFACEDETECTSDK_H
#define GSFACEDETECTSDK_H

#include <iostream>
#include <unistd.h>
#include "public_data.h"

#ifdef _MSC_VER
#ifndef _EXPORT_LIBXLCRACK_DLL_
	#define EXPORT_LIBXLCRACK  _declspec(dllimport)
#else
	#define EXPORT_LIBXLCRACK  _declspec(dllexport)
#endif
#else
#define GS_VISIBILITY __attribute__ ((visibility ("default")))
#endif

using namespace std;
using namespace IODATA;

class MTCNN;
class RetinaFace;
class FeaturExtract;

/*!
 * MultiThread support
 * less OPENCV 2.3.14
 * load more times face module; hardware:NVIDIA and less 1.2G GPU memory
 * Realize face recognition of video frames, face frame coordinates,
 * face feature values, face feature comparison;
 */
class GsFaceDetectSDK
{
public:

   /*!
    *  \brief constuctorSDk
	* \param module path
	* \param MTCNN input arguments. default:{0,40,0.709,{112,112},{0.6,0.7,0.9},{0.5,0.7,0.7},"fc1_output"}
    * \param Whether to use GPU or not.default use GPU
	*/
	GS_VISIBILITY GsFaceDetectSDK(const char* = "./", MtcnnPar* = nullptr, bool = true);

   /*!
    * \brief get face pointer and NUMS
    * \param picture data
    * \param picture width
    * \param picture height
    * \param picture channel. best use 3
    * \param Face coordinate values feature pointer; eg: FaceRect[n]
    * \param Enable multiscale dynamic picture; default enable
    * \param Specify to get the face data type, this version only has 3 ways;
    *        1.FACEALL:all. 2.FACEBOX:only the face frame. 3.FACEFEATURE:features extract.
    *        if used FACEFEATURE mode, must setting FaceRect.score > 0.6 feature will be calculated
    * \return face NUMS; if pixel width or height < 50, pFaceRect->score = 0.0
    */
	GS_VISIBILITY FACE_NUMBERS getFacesAllResult(const uchar*, int, int, int channel,
    		pFaceRect, bool = true, GetFaceType = GetFaceType::FACEALL);

    /*!
     * \brief Reload Interface
     * \param picture data, cv::Mat BGR format
     * \param Face coordinate values feature pointer; eg: FaceRect[n]
     * \param Enable multiscale dynamic picture; default enable
     * \param Specify to get the face data type, this version only has 3 ways;
     *  1.FACEALL:all. 2.FACEBOX:only the face frame. 3.FACEFEATURE:features extract.
     *  \return face NUMS; if pixel width or height < 50, pFaceRect->score = 0.0
     */
	GS_VISIBILITY FACE_NUMBERS getFacesAllResult(cv::Mat, pFaceRect, bool = true, GetFaceType = GetFaceType::FACEALL);

    /* \brief Only get face feature
     * \param picture data, cv::Mat BGR format
     * \param 512 feature array
	 * \return true success and false error
     */
	GS_VISIBILITY bool getFacesFeatureResult(cv::Mat, float*);

    /*!
     * \brief release object allocate memory;
     */
	GS_VISIBILITY void ReleaseResource(int = 0);

    /*!
     * \brief feature compare
     * \param feature1 pointer 512 float data
     * \param feature2 pointer 512 float data
     * \return 0~1 Similarity
     */
	GS_VISIBILITY float compares(const float*, const float*);

    /*!
     * \brief Get the number of GPUs.
     * \param pointer to integer that will hold the number of GPUs available.
     * \return 0 when success, -1 when failure happens.
     */
	GS_VISIBILITY int getGPUCount(int* out);

    /*!
     * \brief Get the version of SDK.
     * \param select false MXNet or default SDK version
     * \return version string
     */
    GS_VISIBILITY const char* getSDKVersion(bool = true);

protected:
    void warpAffineI(cv::Mat&, FaceRect&, std::vector<cv::Mat>&);

    cv::Mat BGRToRGB(cv::Mat&);

    float normalization(cv::Mat&);

private:
    bool m_InitStatus;
    int m_pictureWSize;
    int m_pictureHSize;
    size_t m_OutWidth;
    size_t m_OutHeight;
    MTCNN* m_mtcnn;
    RetinaFace* m_retinaFace;
    FeaturExtract* m_feaExt;
};

#endif // GSFACEDETECTSDK_H
