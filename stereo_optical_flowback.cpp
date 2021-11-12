#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "stereo_lk/OpticalFlowPyrLKDx.h"

using namespace std;
using namespace cv;

// KITTI1_1 KITTI1_2
string file_1 = "../kitti0_l.png";  // first image
string file_2 = "../kitti0_r.png";  // second image

int main()
{
  // images, note they are CV_8UC1, not CV_8UC3
  Mat img1 = imread(file_1, 0);
  Mat img2 = imread(file_2, 0);

  if(img1.empty() || img2.empty())
    cout<<"read image error! \n"<<file_1<<"\n"<<file_2<<endl;

  auto t1 = chrono::steady_clock::now();

  // cv::Ptr<DesctiptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  // key points, using GFTT here.
  vector<KeyPoint> kp1,kp2,kp1_1;
  Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
  detector->detect(img1, kp1);
  detector->detect(img2, kp2);

  // double t1,t2,time_used;
  // use opencv's flow for validation
  vector<Point2f> pt1, pt2, pt1_back;
  for (auto kp: kp1) 
    pt1.push_back(kp.pt);

  vector<uchar> status_forw,status_back;
  vector<float> error_forw,error_back;
  
  // Flow forward
  // cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
  CalcOpticalFlowPyrDx(img1, img2, pt1, pt2, status_forw, error_forw);
  ReduceVector(pt1,status_forw);
  ReduceVector(pt2,status_forw);
  ReduceVector(kp1,status_forw);
  // cout<<"size1:"<<kp1.size()<<" "<<pt1.size()<<endl;

  // Flow back
  CalcOpticalFlowPyrDx(img2, img1, pt2, pt1_back, status_back, error_back);
  ReduceVector(pt2,status_back);
  ReduceVector(pt1_back,status_back);
  ReduceVector(kp1,status_back);
  
  auto t2 = chrono::steady_clock::now();
  auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optical flow by opencv: " << time_used.count() << endl;

  cv::Mat show_points;
  cv::cvtColor(img1, show_points, CV_GRAY2BGR);
  for (int i = 0; i < pt1.size(); i++)
  {
    if (status_forw[i])
    cv::circle(show_points, pt1[i], 2, cv::Scalar(0, 250, 0), 1);
  }
 
  cout<<"pt1_origin_size:"<<pt1.size()<<endl;
  cout<<"pt1_back_size:"<<kp1.size()<<" "<<pt1_back.size()<<endl;
  // cv::findFundamentalMat(pt1,pt2, cv::FM_RANSAC, 1, 0.99, outlier_status);
  // ReduceVector(pt1,outlier_status);
  // ReduceVector(pt2,outlier_status);
  Mat img2_CV;
  cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
  for (int i = 0; i < pt2.size(); i++) 
  {
    if (status_back[i]) 
    {
      cv::circle(img2_CV, pt2[i], 2, cv::Scalar(250, 0 ,0 ), 1);
      cv::line(img2_CV, pt1_back[i], pt2[i], cv::Scalar(0, 250, 0));
    }
  }
  
  // Draw with Munsell Color System
  // ShowMotionColor(img2,0,pt1,pt2);
  cv::imshow("original points", show_points);
  cv::imshow("tracked ", img2_CV);

  // show_stereo_match("show",img1,img2,pt1,pt2);
  cv::waitKey(0);

  return 0;
}
