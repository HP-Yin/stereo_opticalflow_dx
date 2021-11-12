#include "OpticalFlowPyrLKDx.h"


/**
 * @brief 函数作用	对输入图像求微分，并将微分结果存储到dst中，微分模板可见原理
 * @param src	待处理图像，src的depth必须为CV_8U,通道不做要求
 * @param dst 存放微分结果，类型同src，通道数为src的2倍
*/
static void ComputeSharrDeriv(const cv::Mat& src, cv::Mat& dst)
{
    int rows = src.rows, 
        cols = src.cols, 
        cn = src.channels(), 
        colsn = cols*cn, 
        depth = src.depth();

    int x, y, delta = (int)alignSize((cols + 2)*cn, 16);
    AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);

    for( y = 0; y < rows; y++ )
    {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        deriv_type* drow = dst.ptr<deriv_type>(y);

        // do vertical convolution
        x = 0;
        for( ; x < colsn; x++ )
        {
            int t0 = (srow0[x] + srow2[x])*3 + srow1[x]*10;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        int x0 = (cols > 1 ? 1 : 0)*cn, x1 = (cols > 1 ? cols-2 : 0)*cn;
        for( int k = 0; k < cn; k++ )
        {
            trow0[-cn + k] = trow0[x0 + k]; trow0[colsn + k] = trow0[x1 + k];
            trow1[-cn + k] = trow1[x0 + k]; trow1[colsn + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;
        for( ; x < colsn; x++ )
        {
            deriv_type t0 = (deriv_type)(trow0[x+cn] - trow0[x-cn]);
            deriv_type t1 = (deriv_type)((trow1[x+cn] + trow1[x-cn])*3 + trow1[x]*10);
            drow[x*2] = t0; drow[x*2+1] = t1;
        }
    }
}

LKTracker::LKTracker( const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                      const Point2f* _prevPts, Point2f* _nextPts,
                      uchar* _status, float* _err,
                      Size _winSize, cv::TermCriteria _criteria,
                      int _level, int _maxLevel, int _flags, float _minEigThreshold )
{
    prevImg = &_prevImg;
    prevDeriv = &_prevDeriv;
    nextImg = &_nextImg;
    prevPts = _prevPts;
    nextPts = _nextPts;
    status = _status;
    err = _err;
    winSize = _winSize;
    criteria = _criteria;
    level = _level;
    maxLevel = _maxLevel;
    flags = _flags;
    minEigThreshold = _minEigThreshold;
}

void LKTracker::operator()(const Range& range) const
{
//设置初值，I,J,derivI,halfWin:半个窗口大小
    Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
    const Mat& I = *prevImg;
    const Mat& J = *nextImg;
    const Mat& derivI = *prevDeriv;

    int j, cn = I.channels(), cn2 = cn*2;
    //通过追踪可知：deriv_type为short类型；_buf为3倍的窗口面积大小
    cv::AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2));
    int derivDepth = DataType<deriv_type>::depth;

    //IWinBuf占据了_buf的第一个窗口面积大小的区域
    cv::Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf);
    //derivIWinBuf占据了_buf后两个窗口面积大小的区域
    cv::Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), (deriv_type*)_buf + winSize.area()*cn);
    // cout<<"CORE DUMP ERR "<<endl;
//挨个遍历各个特征点，ptidx为特征点的索引
    for( int ptidx = range.start; ptidx < range.end; ptidx++ )
    {
        // cout<<"range.start: "<<range.start<<endl;
        //将特征点的坐标映射到当前层来
        cv::Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level)); // 0.125 0.25 0.5 1
        // cout<<"prevPt: "<<prevPt<<endl;
        cv::Point2f nextPt;

        //最高层金字塔，令nextPt的坐标点等同于prevPt
        if( level == maxLevel )
        {
            // if( flags & OPTFLOW_USE_INITIAL_FLOW )
                // nextPt = nextPts[ptidx]*(float)(1./(1 << level));
            // else
                nextPt = prevPt;
        }
        //非最高层，令nextPt的坐标*2，高层坐标反映射到低一层坐标上
        else
            nextPt = nextPts[ptidx]*2.f;

        nextPts[ptidx] = nextPt;

        //对浮点型的prevPt取整和取浮点计算
        Point2i iprevPt, inextPt;
        //对当前层特征点坐标向左上移动halfWin个单位，为了计算模板起始位置。
        //特征prevPt的G值的计算是通过以prevPt为中心，大小为winSize的模板，因此模板起始点为prevPt-halfWin。
        prevPt -= halfWin;
        iprevPt.x = cvFloor(prevPt.x);
        iprevPt.y = cvFloor(prevPt.y);

        //坐标点出界判断
        if( iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
            iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows )
        {
            if( level == 0 )
            {
                if( status )
                    status[ptidx] = false;
                if( err )
                    err[ptidx] = 0;
            }
            continue;
        }

        //根据浮点数值，计算四个坐标二次线性插值的系数
        float a = prevPt.x - iprevPt.x; //       
        float b = prevPt.y - iprevPt.y;
        const int W_BITS = 14, W_BITS1 = 14;
        const float FLT_SCALE = 1.f/(1 << 20);
        //opencv中，为了高效的计算效率，对浮点数乘以大数转为整形进行计算
        int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
        int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
        int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
        int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
        //eleSize1:一个元素占据的字节数
        //step:一行元素占据的字节数
        int dstep = (int)(derivI.step/derivI.elemSize1());
        int stepI = (int)(I.step/I.elemSize1());
        int stepJ = (int)(J.step/J.elemSize1());
        acctype iA11 = 0, iA12 = 0, iA22 = 0;
        float A11, A12, A22;

        //计算矩阵G的三个元素Ix*Ix，Iy*Iy,Ix*Iy
        // extract the patch from the first image, compute covariation matrix of derivatives
        int x, y;
        for( y = 0; y < winSize.height; y++ )
        {
            const uchar* src = I.ptr() + (y + iprevPt.y)*stepI + iprevPt.x*cn;
            const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y)*dstep + iprevPt.x*cn2;

            deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
            deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

            x = 0;
            for( ; x < winSize.width*cn; x++, dsrc += 2, dIptr += 2 )
            {
                int ival =  CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 + src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, 
                                       W_BITS1-5);
                int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 + dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, 
                                       W_BITS1);
                int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 + dsrc[dstep+cn2+1]*iw11, 
                                       W_BITS1);

                Iptr[x] = (short)ival;  //线性插值后的像素灰度值
                dIptr[0] = (short)ixval;
                dIptr[1] = (short)iyval;

                iA11 += (itemtype)(ixval*ixval);
                iA12 += (itemtype)(ixval*iyval);
                iA22 += (itemtype)(iyval*iyval);
            }
        }

        //对G中的三个元素恢复到原来的尺度
        A11 = iA11*FLT_SCALE;
        A12 = iA12*FLT_SCALE;
        A22 = iA22*FLT_SCALE;
        //D计算的是G的行列式值
        float D = A11*A22 - A12*A12;
        //minEig计算的是一元二次方程的根
        float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2*winSize.width*winSize.height);
        if( err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0 )
            err[ptidx] = (float)minEig;

        //若行列式值==0，则G可逆；若根值小于某个固定值，则有根，有根矩阵可逆
        if( minEig < minEigThreshold || D < FLT_EPSILON )
        {
            if( level == 0 && status )
                status[ptidx] = false;
            continue;
        }

        D = 1.f/D;

        //nextPt也移动至模板开始的位置
        nextPt -= halfWin;
        Point2f prevDelta;

        //对特征点进行迭代计算
        for( j = 0; j < criteria.maxCount; j++ )
        {
            //先对图像J上的特征点位置取整并进行越界判断
            inextPt.x = cvFloor(nextPt.x);
            inextPt.y = cvFloor(nextPt.y);
            if( inextPt.x < -winSize.width || inextPt.x >= J.cols ||
               inextPt.y < -winSize.height || inextPt.y >= J.rows )
            {
                if( level == 0 && status )
                    status[ptidx] = false;
                break;
            }

            //求解四个特征点的二次线性插值系数
            a = nextPt.x - inextPt.x;
            b = nextPt.y - inextPt.y;
            iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            acctype ib1 = 0, ib2 = 0;
            float b1, b2;

            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPt.y)*stepJ + inextPt.x*cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                const deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);
                x = 0;
                for( ; x < winSize.width*cn; x++, dIptr += 2 )
                {
                    //I,J两个特征点求模板范围内的累计差值
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    //计算矩阵b
                    ib1 += (itemtype)(diff*dIptr[0]);
                    ib2 += (itemtype)(diff*dIptr[1]);
                }
            }
            b1 = ib1*FLT_SCALE;
            b2 = ib2*FLT_SCALE;

            //根据G的逆*b得到特征点的速度v（delta）
            // TODO: 这里的y方向不更新
            Point2f delta( (float)((A12*b2 - A22*b1) * D),
                           0);
            // Point2f delta( (float)((A12*b2 - A22*b1) * D),
            //                (float)((A12*b1 - A11*b2) * D));
            //delta = -delta;

            //给J中的特征点叠加一个偏移delta
            nextPt += delta;
            //将更新后的位置坐标保存起来
            nextPts[ptidx] = nextPt + halfWin;
            //当delta偏移值很小时，可跳出迭代，认为该delta就是最终的delta
            if( delta.ddot(delta) <= criteria.epsilon )
                break;
            //如果前后两次delta变化比较小，也可认为已经找到delta了
            if( j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
               std::abs(delta.y + prevDelta.y) < 0.01 )
            {
                nextPts[ptidx] -= delta*0.5f;
                break;
            }
            prevDelta = delta;
        }

//根据光流法求最优delta，保证图像I,J之间的特征点的误差值最小。程序到了这一环节，光流法求特征点已经计算完成
//第三部分--计算误差    
        CV_Assert(status != NULL);
        if( status[ptidx] && err && level == 0 && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0 )
        {
            Point2f nextPoint = nextPts[ptidx] - halfWin;
            Point inextPoint;

            inextPoint.x = cvFloor(nextPoint.x);
            inextPoint.y = cvFloor(nextPoint.y);

            if( inextPoint.x < -winSize.width || inextPoint.x >= J.cols ||
                inextPoint.y < -winSize.height || inextPoint.y >= J.rows )
            {
                if( status )
                    status[ptidx] = false;
                continue;
            }

            float aa = nextPoint.x - inextPoint.x;
            float bb = nextPoint.y - inextPoint.y;
            iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
            iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
            iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float errval = 0.f;

            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPoint.y)*stepJ + inextPoint.x*cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);

                for( x = 0; x < winSize.width*cn; x++ )
                {
                    //特征点在I、J上的误差累计值
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    errval += std::abs((float)diff);
                }
            }
            //得到最终的err，这个值可根据自己经验去使用，是衡量两个特征点的误差指标
            err[ptidx] = errval * 1.f/(32*winSize.width*cn*winSize.height);
        }
    }
}

void CalcOpticalFlowPyrDx(const cv::Mat img1,const cv::Mat img2,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err)
{
    cv::TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

    cv::Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<deriv_type>::depth;

    int i,npoints;
    // npoints = prev_pts.size();
    CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);//checkVector检查这个Mat是否为Vector
    // cout<<"npoints "<<npoints<<endl;

    //特征点数量判断
    if( npoints == 0 )
    {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    //变量初始化操作
    // if( !(flags & OPTFLOW_USE_INITIAL_FLOW) )
        // _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);
    cv::Mat nextPtsMat = _nextPts.getMat(); //
    // cout<<"Next npoints "<<nextPtsMat.checkVector(2, CV_32F, true)<<endl; // -1
    CV_Assert( nextPtsMat.checkVector(2, CV_32F, true) == npoints );

    const cv::Point2f* prevPts = prevPtsMat.ptr<Point2f>();
          cv::Point2f* nextPts = nextPtsMat.ptr<Point2f>();
    if(nextPts==nullptr)
        cout<<"nextPts==nullptr"<<endl;

    _status.create((int)npoints, 1, CV_8U, -1, true);
    cv::Mat statusMat = _status.getMat();
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.ptr();
    cv::Mat errMat;
    float* err = 0;

    // CV_Assert(npoints = next_pts.size());

    for( i = 0; i < npoints; i++ )
        status[i] = true;

    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = errMat.ptr<float>();
    }


    cv::Size win_size(21,21);
    int max_level = 3;
    // const int derivDepth = 3;

    vector<cv::Mat> prev_pyr,next_pyr;
    int levels1 = -1;
    int lvlStep1 = 1;
    int levels2 = -1;
    int lvlStep2 = 1;

    max_level = cv::buildOpticalFlowPyramid(img1,prev_pyr,win_size,max_level,false);
    max_level = cv::buildOpticalFlowPyramid(img2,next_pyr,win_size,max_level,false);
    // cv::imshow("prev_pyr[2]",prev_pyr[3]);
    // cv::waitKey(0);

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;
    
    // dI/dx ~ Ix, dI/dy ~ Iy
    cv::Mat derivIBuf;
    if(lvlStep1 == 1)
        derivIBuf.create(prev_pyr[0].rows + win_size.height*2, 
                         prev_pyr[0].cols + win_size.width*2, 
                         CV_MAKETYPE(derivDepth, prev_pyr[0].channels() * 2));

    for(int level = max_level; level >= 0; level-- )
    {
        //存放每一层图像的微分图
        cv::Mat derivI;
        //一般情况下lvlStep1是=1的
        if(lvlStep1 == 1)
        {
            //获取当前层图像的size
            cv::Size imgSize = prev_pyr[level * lvlStep1].size();
            //
            //定义一个_derivI（和derivBuf共享内存），高为：imgSize.height + winSize.height*2
            //宽为：imgSize.width + winSize.width*2
            cv::Mat _derivI(imgSize.height + win_size.height*2,
                            imgSize.width  + win_size.width*2, derivIBuf.type(), derivIBuf.ptr() );
            //derivI与_derivI共享内存，derivI从_deriveI的起点（winSize.width, winSize.height）开始，
            //derivI宽度为imgSize.width,高度为imgSize.height
            derivI = _derivI(Rect(win_size.width, win_size.height, imgSize.width, imgSize.height));
            //将当前层的图像求取微分，存放在derivI中
            ComputeSharrDeriv(prev_pyr[level * lvlStep1], derivI);
            // cout<<"derivI.channels(): "<<derivI.channels()<<endl;
            // cv::waitKey(0);
            copyMakeBorder(derivI, _derivI, win_size.height, win_size.height, 
                                            win_size.width, win_size.width, 
                                            BORDER_CONSTANT|BORDER_ISOLATED);
            // cv::imshow("_derivI",_derivI);
            // cv::waitKey(0);
        }
        else
        {}
        //两帧金字塔图像尺寸对比
        CV_Assert(prev_pyr[level * lvlStep1].size() == next_pyr[level * lvlStep2].size());
        CV_Assert(prev_pyr[level * lvlStep1].type() == next_pyr[level * lvlStep2].type());
        
        typedef LKTracker LKTracker;
        //实质进行光流算法的主体，LKTrackerInvoker（），该函数中完成了光流算法的具体操作
        cv::parallel_for_(Range(0, npoints), LKTracker(prev_pyr[level * lvlStep1], derivI, 
                                                       next_pyr[level * lvlStep2], 
                                                       prevPts, nextPts, status, err, win_size,
                                                       criteria,level, max_level, 0, 0.0001) );                                                    
    }
    // cv::imshow("prev_pyr[3]",prev_pyr[3]);
    // cv::waitKey(0);
}

void CalcOpticalFlowOneLevel(const cv::Mat img1,const cv::Mat img2,
                             InputArray _prevPts, InputOutputArray _nextPts,
                             OutputArray _status, OutputArray _err)
{
    cv::TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

    cv::Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<deriv_type>::depth;

    int i,npoints;
    // npoints = prev_pts.size();
    CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);//checkVector检查这个Mat是否为Vector
    // cout<<"npoints "<<npoints<<endl;

    //特征点数量判断
    if( npoints == 0 )
    {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    //变量初始化操作
    // if( !(flags & OPTFLOW_USE_INITIAL_FLOW) )
        // _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);
    cv::Mat nextPtsMat = _nextPts.getMat(); //
    // cout<<"Next npoints "<<nextPtsMat.checkVector(2, CV_32F, true)<<endl; // -1
    CV_Assert( nextPtsMat.checkVector(2, CV_32F, true) == npoints );

    const cv::Point2f* prevPts = prevPtsMat.ptr<Point2f>();
          cv::Point2f* nextPts = nextPtsMat.ptr<Point2f>();
    if(nextPts==nullptr)
        cout<<"nextPts==nullptr"<<endl;

    _status.create((int)npoints, 1, CV_8U, -1, true);
    cv::Mat statusMat = _status.getMat();
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.ptr();
    cv::Mat errMat;
    float* err = 0;

    // CV_Assert(npoints = next_pts.size());

    for( i = 0; i < npoints; i++ )
        status[i] = true;

    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = errMat.ptr<float>();
    }

    cv::Size win_size(21,21);
    // int max_level = 3;
    // const int derivDepth = 3;

    cv::Mat prev_img = img1.clone(),
            next_img = img2.clone();
    int levels1 = -1;
    int lvlStep1 = 1;
    int levels2 = -1;
    int lvlStep2 = 1;

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;
    
    // dI/dx ~ Ix, dI/dy ~ Iy
    cv::Mat derivIBuf;
    if(lvlStep1 == 1)
        derivIBuf.create(prev_img.rows + win_size.height*2, 
                         prev_img.cols + win_size.width*2, 
                         CV_MAKETYPE(derivDepth, prev_img.channels() * 2));

        //存放每一层图像的微分图
        cv::Mat derivI;
        //一般情况下lvlStep1是=1的
        if(lvlStep1 == 1)
        {
            //获取当前层图像的size
            cv::Size imgSize = prev_img.size();
            //
            //定义一个_derivI（和derivBuf共享内存），高为：imgSize.height + winSize.height*2
            //宽为：imgSize.width + winSize.width*2
            cv::Mat _derivI(imgSize.height + win_size.height*2,
                            imgSize.width  + win_size.width*2, derivIBuf.type(), derivIBuf.ptr() );
            //derivI与_derivI共享内存，derivI从_deriveI的起点（winSize.width, winSize.height）开始，
            //derivI宽度为imgSize.width,高度为imgSize.height
            derivI = _derivI(Rect(win_size.width, win_size.height, imgSize.width, imgSize.height));
            //将当前层的图像求取微分，存放在derivI中
            ComputeSharrDeriv(prev_img, derivI);
            // cout<<"derivI.channels(): "<<derivI.channels()<<endl;
            // cv::waitKey(0);
            copyMakeBorder(derivI, _derivI, win_size.height, win_size.height, 
                                            win_size.width, win_size.width, 
                                            BORDER_CONSTANT|BORDER_ISOLATED);
            // cv::imshow("_derivI",_derivI);
            // cv::waitKey(0);
        }
        else
        {}
        //两帧金字塔图像尺寸对比
        CV_Assert(prev_img.size() == next_img.size());
        CV_Assert(prev_img.type() == next_img.type());
        
        typedef LKTracker LKTracker;
        //实质进行光流算法的主体，LKTrackerInvoker（），该函数中完成了光流算法的具体操作
        cv::parallel_for_(Range(0, npoints), LKTracker(prev_img, derivI, 
                                                       next_img, 
                                                       prevPts, nextPts, status, err, win_size,
                                                       criteria,0, 0, 0, 0.0001) );                             
}