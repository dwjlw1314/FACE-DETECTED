#ifndef MTCNN_NET_H
#define MTCNN_NET_H

/*
 * The switch of image value selection mode. can increase the running speed
 * and reduce the detection accuracy after it is turned on.
 */
//#define INTER_FAST 1

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "mxnet-cpp/MxNetCpp.h"
#include "public_data.h"

using namespace std;
using namespace mxnet::cpp;
using namespace IODATA;

class NetStruct
{
public:
    NetStruct();
    ~NetStruct();

    void Forward(NDArray *input = nullptr);

    void InitMember(const std::string, const std::string, Shape);

    void SetThreshold(const mx_float);
    mx_float GetThreshold() const;

    void SetNmsThreshold(const mx_float);
    mx_float GetNmsThreshold() const;

    NDArray& InputDataNDArray();
    const NDArray* OutputNDArray(size_t);

    const std::vector<mx_uint> GetDataShape();

    void GetExecutorNDarray();
    void DelExecutorNDarray();

private:
    /*Fill the trained parameter into the model, a.k.a. net, executor*/
    void LoadParameters(const std::string&);

    mx_float m_threshold;
    mx_float m_nms_threshold;

    Symbol m_net;
    Executor* m_executor;

    map<std::string, NDArray> m_args_map;
    map<std::string, NDArray> m_aux_map;
    std::vector<NDArray*> m_out_ndarray;
};

class MTCNN
{
public:
    MTCNN(const char*, MtcnnPar *, bool);

    ~MTCNN();

    void Detect(const cv::Mat&, std::vector<FaceRect>&);

private:

    void SetParameter(const MtcnnPar*);

    void InitMtcnn(const std::string&);

    bool CvMatToNDArraySignalChannel(const cv::Mat&, NDArray*);

    void Preprocess(const cv::Mat&, std::vector<cv::Mat>*);

    /*
     * Normalize the pictures unused
     */
    void SetMean(NDArray&);

    void Padding(int, int);

    /*
     * GPU model unused, cancel cv::split()
     */
    void WrapInputLayer(std::vector<cv::Mat>*, const NDArray*, const int, const int);

    void GenerateBoundingBox(const NDArray*, const NDArray*, double, const mx_float, int, int);

    void ClassifyFace(const std::vector<FaceRect>&, cv::Mat&, NetStruct*, mx_float, char);

    void ClassifyFace_MulImage(const std::vector<FaceRect>&, cv::Mat&, NetStruct*, mx_float, char);

    void Bbox2Square(std::vector<FaceRect>&);

    void RegressPoint(const std::vector<FaceRect>&);

    void AddNDArrayVector(NDArray*, std::vector<NDArray>);

    std::vector<FaceRect> BoxRegress(std::vector<FaceRect>&, int);

    std::vector<FaceRect> NonMaximumSuppression(std::vector<FaceRect>&, const mx_float, char);

private:
    int m_minSize;
    mx_float m_factor;

    NetStruct * m_PNet;
    NetStruct * m_RNet;
    NetStruct * m_ONet;

    std::vector<FaceRect> m_condidate_rects;
    std::vector<FaceRect> m_total_boxes;
    std::vector<FaceRect> m_regressed_rects;
    std::vector<FaceRect> m_regressed_pading;
};

#endif // MTCNN_NET_H
