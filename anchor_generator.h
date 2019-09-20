#ifndef ANCHOR_GENERTOR
#define ANCHOR_GENERTOR

#include <vector>
#include <iostream>
#include "mxnet-cpp/MxNetCpp.h"
#include "public_data.h"

using namespace std;
using namespace mxnet::cpp;

struct AnchorCfg
{
	std::vector<float> SCALES;
	std::vector<float> RATIOS;
	int BASE_SIZE;
};

class CRect2f
{
public:
    CRect2f(float, float, float, float);
    float& operator[](int);
    float operator[](int) const;

    float val[4];
};

class Anchor
{
public:
    bool operator<(const Anchor&) const;
    bool operator>(const Anchor&) const;

    float operator[](int) const;

	float reg[4]; //offset reg
	float score; //cls score

    cv::Rect_< float > anchor; //x1,y1,x2,y2
    cv::Rect_< float > finalbox; //final box res

    cv::Point center; //anchor feat center
    std::vector<cv::Point2f> pts; //landmarks
};

class AnchorGenerator
{
public:
	AnchorGenerator();
	~AnchorGenerator();

    // init different anchors
    int Init(int, float, const AnchorCfg&, bool);

	// filter anchors and return valid anchors
    int FilterAnchor(NDArray&, NDArray&, NDArray&, vector<Anchor>&);

private:
    void _ratio_enum(const CRect2f&, const std::vector<float>&, std::vector<CRect2f>&);

    void _scale_enum(const vector<CRect2f>&, const vector<float>&, vector<CRect2f>&);

    void bbox_pred(const CRect2f&, const CRect2f&, cv::Rect_<float>&);

    void landmark_pred(const CRect2f, const vector<cv::Point2f>&, vector<cv::Point2f>&);

	int anchor_stride; //anchor tile stride
	mx_uint anchor_num; //anchor type num
    std::vector<CRect2f> preset_anchors;
    float cls_threshold;
};

#endif // ANCHOR_GENERTOR
