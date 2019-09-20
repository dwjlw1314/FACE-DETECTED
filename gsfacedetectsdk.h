#ifndef GSFACEDETECTSDK_H
#define GSFACEDETECTSDK_H

#include <iostream>
#include "public_data.h"

using namespace std;
using namespace IODATA;

class MTCNN;
class RetinaFace;
class FeaturExtract;

/*!
 * Multithread support
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
	GsFaceDetectSDK(const char* = "./", MtcnnPar* = nullptr, bool = true);

   /*!
    * \brief get face pointer and nums
    * \param picture data
    * \param picture width
    * \param picture height
    * \param picture channel. best use 3
    * \param Face coordinate values feature pointer; eg: FaceRect[n]
    * \param Specify to get the face data type, this version only has 3 ways;
    *        1.FACEALL:all. 2.FACEBOX:only the face frame. 3.FACEFEATURE:features extract.
    * \return face nums; if pixel width or heigth < 50, pFaceRect->score = 0.0
    */
    FACE_NUMBERS getFacesAllResult(const uchar*, int, int, int channel,
    		pFaceRect, GetFaceType = GetFaceType::FACEALL);

    /*!
     * \brief Reload Interface
     * \param picture data, cv::Mat BGR format
     * \param Face coordinate values feature pointer; eg: FaceRect[n]
     * \param Specify to get the face data type, this version only has 3 ways;
     *  1.FACEALL:all. 2.FACEBOX:only the face frame. 3.FACEFEATURE:features extract.
     *  \return face nums; if pixel width or heigth < 50, pFaceRect->score = 0.0
     */
    FACE_NUMBERS getFacesAllResult(cv::Mat, pFaceRect, GetFaceType = GetFaceType::FACEALL);

    /* \brief Only get face feature
     * \param picture data, cv::Mat BGR format
     * \param 512 feature array
	 * \return true success and false error
     */
    bool getFacesFeatureResult(cv::Mat, float*);

    /*!
     * \brief release object allocate memory;
     */
    void ReleaseResource(void);

    /*!
     * \brief feature compare
     * \param feature1 pointer 512 float data
     * \param feature2 pointer 512 float data
     * \return 0~1 Similarity
     */
    float compares(const float*, const float*);

    /*!
     * \brief Get the number of GPUs.
     * \param pointer to int that will hold the number of GPUs available.
     * \return 0 when success, -1 when failure happens.
     */
    int getGPUCount(int* out);

    /*!
     * \brief Get the version of sdk.
     * \return version string
     */
    const char* getSDKVersion();

protected:
    void warpAffineI(cv::Mat&, FaceRect&, std::vector<cv::Mat>&);

    cv::Mat BGRToRGB(cv::Mat&);

private:
    size_t m_OutWidth;
    size_t m_OutHeight;
    MTCNN* m_mtcnn;
    RetinaFace* m_retinaFace;
    FeaturExtract* m_feaExt;
};

#endif // GSFACEDETECTSDK_H
