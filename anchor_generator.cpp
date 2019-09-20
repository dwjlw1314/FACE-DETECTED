#include "anchor_generator.h"

CRect2f::CRect2f(float x1, float y1, float x2, float y2)
{
	val[0] = x1;
	val[1] = y1;
	val[2] = x2;
	val[3] = y2;
}

float& CRect2f::operator[](int i)
{
    return val[i];
}

float CRect2f::operator[](int i) const
{
    return val[i];
}

bool Anchor::operator<(const Anchor &t) const
{
    return score < t.score;
}

bool Anchor::operator>(const Anchor &t) const
{
    return score > t.score;
}

float Anchor::operator[](int i) const
{
	if (i == 0)
		return finalbox.x;
	if (i == 1)
		return finalbox.y;
	if (i == 2)
		return finalbox.width;
	if (i == 3)
		return finalbox.height;

	return 0.0;
}

AnchorGenerator::AnchorGenerator()
{
	anchor_stride = 0;
	anchor_num = 2;
    cls_threshold = 0.8;
}

AnchorGenerator::~AnchorGenerator() {}

// init different anchors
int AnchorGenerator::Init(int stride, float threshold, const AnchorCfg& cfg, bool dense_anchor)
{
	CRect2f base_anchor(0, 0, cfg.BASE_SIZE-1, cfg.BASE_SIZE-1);
	std::vector<CRect2f> ratio_anchors;
	// get ratio anchors
	_ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
	_scale_enum(ratio_anchors, cfg.SCALES, preset_anchors);

	// save as x1,y1,x2,y2
	if (dense_anchor)
	{
		assert(stride % 2 == 0);
		int num = preset_anchors.size();
		for (int i = 0; i < num; ++i)
		{
			CRect2f anchor = preset_anchors[i];
			preset_anchors.push_back(CRect2f(anchor[0]+int(stride/2), anchor[1]+int(stride/2),
					anchor[2]+int(stride/2), anchor[3]+int(stride/2)));
		}
	}

    anchor_stride = stride;
	cls_threshold = threshold;
	anchor_num = preset_anchors.size();

	return anchor_num;
}


int AnchorGenerator::FilterAnchor(NDArray& cls, NDArray& reg, NDArray& pts, vector<Anchor>& result)
{
    if(cls.GetShape()[1] != anchor_num*2) { return -1; }
    if(reg.GetShape()[1] != anchor_num*4) { return -1; }
	if((pts.GetShape()[1] % anchor_num) != 0) { return -1; }

	int pts_length = pts.GetShape()[1]/anchor_num/2;

    int w = cls.GetShape()[3];
    int h = cls.GetShape()[2];

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
        	int id = i * w + j;
            for (mx_uint a = 0; a < anchor_num; ++a)
            {
            	//std::cout<< j << "--------"<< i << "--------"<< id << "--------"<<cls.channel(anchor_num + a)[id]<<std::endl;
            	if (cls.GetData()[(anchor_num + a) * h * w + id] >= cls_threshold)
            	{
                    //printf("cls %f\n", cls.channel(anchor_num + a)[id]);
                    CRect2f box(j * anchor_stride + preset_anchors[a][0],
                            i * anchor_stride + preset_anchors[a][1],
                            j * anchor_stride + preset_anchors[a][2],
                            i * anchor_stride + preset_anchors[a][3]);
                    //printf("%f %f %f %f\n", box[0], box[1], box[2], box[3]);
                    CRect2f delta(reg.GetData()[(a*4+0) * h * w+id],
                    		reg.GetData()[(a*4+1) * h * w+id],
							reg.GetData()[(a*4+2) * h * w+id],
							reg.GetData()[(a*4+3) * h * w+id]);

                    Anchor res;
                    res.anchor = cv::Rect_< float >(box[0], box[1], box[2], box[3]);
                    bbox_pred(box, delta, res.finalbox);
                    //printf("bbox pred\n");
                    res.score = cls.GetData()[(anchor_num + a) * h * w+id];
                    res.center = cv::Point(j,i);

					std::vector<cv::Point2f> pts_delta(pts_length);
					for (int p = 0; p < pts_length; ++p)
					{
						pts_delta[p].x = pts.GetData()[(a*pts_length*2+p*2) * h * w+id];
						pts_delta[p].y = pts.GetData()[(a*pts_length*2+p*2+1) * h * w+id];
					}

					landmark_pred(box, pts_delta, res.pts);

                    result.push_back(res);
                }
            }
        }
    }
	return 0;
}

void AnchorGenerator::_ratio_enum(const CRect2f& anchor, const vector<float>& ratios, vector<CRect2f>& ratio_anchors)
{
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

	ratio_anchors.clear();
	float sz = w * h;
	for (size_t s = 0; s < ratios.size(); ++s)
	{
		float r = ratios[s];
		float size_ratios = sz / r;
		float ws = std::sqrt(size_ratios);
		float hs = ws * r;
		ratio_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
				y_ctr - 0.5 * (hs - 1),
				x_ctr + 0.5 * (ws - 1),
				y_ctr + 0.5 * (hs - 1)));
	}
}

void AnchorGenerator::_scale_enum(const vector<CRect2f>& ratio_anchor, const vector<float>& scales, vector<CRect2f>& scale_anchors)
{
	scale_anchors.clear();
	for (size_t a = 0; a < ratio_anchor.size(); ++a)
	{
		CRect2f anchor = ratio_anchor[a];
		float w = anchor[2] - anchor[0] + 1;	
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		for (size_t s = 0; s < scales.size(); ++s)
		{
			float ws = w * scales[s];
			float hs = h * scales[s];
			scale_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
				y_ctr - 0.5 * (hs - 1),
				x_ctr + 0.5 * (ws - 1),
				y_ctr + 0.5 * (hs - 1)));
		}
	}
}

void AnchorGenerator::bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect_< float >& box)
{
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

    float dx = delta[0];
    float dy = delta[1];
    float dw = delta[2];
    float dh = delta[3];

    float pred_ctr_x = dx * w + x_ctr; 
    float pred_ctr_y = dy * h + y_ctr;
    float pred_w = std::exp(dw) * w; 
    float pred_h = std::exp(dh) * h;

    box = cv::Rect_< float >(pred_ctr_x - 0.5 * (pred_w - 1.0),
		pred_ctr_y - 0.5 * (pred_h - 1.0),
		pred_ctr_x + 0.5 * (pred_w - 1.0),
		pred_ctr_y + 0.5 * (pred_h - 1.0));
}

void AnchorGenerator::landmark_pred(const CRect2f anchor, const vector<cv::Point2f>& delta, vector<cv::Point2f>& pts)
{
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

    pts.resize(delta.size());
    for (size_t i = 0; i < delta.size(); ++i)
    {
        pts[i].x = delta[i].x*w + x_ctr;
        pts[i].y = delta[i].y*h + y_ctr;
    }
}
