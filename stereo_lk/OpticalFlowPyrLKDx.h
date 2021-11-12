#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 用宏定义一个函数 (x+(n-1)*2) / 2^n
#define  CV_DESCALE(x,n)     ( ( (x) + (1 << ((n)-1)) ) >> (n) )

typedef short deriv_type;
typedef float acctype;
typedef float itemtype;
struct LKTracker : cv::ParallelLoopBody
{
    LKTracker(const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                const Point2f* _prevPts, Point2f* _nextPts,
                uchar* _status, float* _err,Size _winSize, TermCriteria _criteria,
                int _level, int _maxLevel, int _flags, float _minEigThreshold );

    void operator()(const Range& range) const;

    const cv::Mat* prevImg;
    const cv::Mat* nextImg;
    const cv::Mat* prevDeriv;
    const cv::Point2f* prevPts;
    cv::Point2f* nextPts;
    uchar* status;
    float* err;
    cv::Size winSize;
    cv::TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;
};

void CalcOpticalFlowPyrDx(const cv::Mat img1,const cv::Mat img2,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err);

void CalcOpticalFlowOneLevel(const cv::Mat img1,const cv::Mat img2,
                             InputArray _prevPts, InputOutputArray _nextPts,
                             OutputArray _status, OutputArray _err);

inline void ReduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
inline void ReduceVector(vector<cv::KeyPoint> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}    
