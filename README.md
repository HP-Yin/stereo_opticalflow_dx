# stereo_opticalflow_dx
calcOpticalFlowPyrLK from opencv, compute transverse only

## Usage:
```
mkdir build
cd build
cmake ..
make -j
./stereo_optical_flowback
```

Set the path of your image at stereo_optical_flowback.cpp
```
// KITTI1_1 KITTI1_2
string file_1 = "../kitti0_l.png";  // first image
string file_2 = "../kitti0_r.png";  // second image
```
