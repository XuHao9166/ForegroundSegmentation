#include < stdio.h>
#include < iostream>

#include < opencv2\opencv.hpp>
#include < opencv2/core/core.hpp>
#include < opencv2/highgui/highgui.hpp>
#include < opencv2/video/background_segm.hpp>
//#include < opencv2\gpu\gpu.hpp>


#include <cuda_runtime.h>  
#include <cuda.h> 

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp> 
#include <opencv2/cudalegacy.hpp>

#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp> 
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp> 

using namespace cv;
using namespace std;

#define UNKNOWN_FLOW_THRESH 1e9  



////////////定义孟塞尔颜色系统来进行彩色图像显示
void makecolorwheel(vector<Scalar> &colorwheel)  //定义孟塞尔颜色系统
{
	int RY = 15;  //红黄
	int YG = 6;  //黄绿
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void motionToColor(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel;
	if (colorwheel.empty())
		makecolorwheel(colorwheel); //定义孟塞尔颜色系统

	
	float maxrad = -1;

	
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
	 

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // 用半径增加饱和度
				else
					col *= .75; //超出范围
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}
/////////////////////////////////////////


 void drawOptFlowMap_gpu(const Mat& flow_x, const Mat& flow_y, Mat& cflowmap, int step, const Scalar& color) {



	 for (int y = 0; y < cflowmap.rows; y += step)
	 for (int x = 0; x < cflowmap.cols; x += step)
	 {
		 Point2f fxy;
		 fxy.x = cvRound(flow_x.at< float >(y, x) + x);
		 fxy.y = cvRound(flow_y.at< float >(y, x) + y);

		 line(cflowmap, Point(x, y), Point(fxy.x, fxy.y), color);
		 circle(cflowmap, Point(fxy.x, fxy.y), 1, color, -1);
	 }
 }

 void showFlow(const char* name, const cv::cuda::GpuMat& d_flow, int col,int row ,int framenum)
{

	 Mat motion2color;
	 Mat flow(d_flow);
	 motionToColor(flow, motion2color);//将视频检测出来的光流转化成颜色显示
	 //resize(flow, flow, Size(340, 256));
	 imshow("彩色光流", motion2color);



	 Mat qianjing(Size(col, row),CV_8UC3);
	 static cv::cuda::GpuMat planes[2];
	cuda::split(d_flow, planes);

	Mat flowx(planes[0]); ////对函数输出的双通道光流图像进行分解
	Mat flowy(planes[1]);



	//resize(opticalFlowOut, opticalFlowOut,Size(col, row));
	drawOptFlowMap_gpu(flowx, flowy, qianjing, 10, CV_RGB(0, 255, 0)); ///输入稠密光流得到的结果输出前景图像

	

	/*char outputname[100];
	sprintf(outputname, "../y/y%d.jpg", framenum);*/

	resize(flowx, flowx, Size(340, 256));
	resize(flowy, flowy, Size(340, 256));

	//imwrite(outputname,  flowx);
	//imwrite(outputname, flowy);
	imshow("x", flowx);
	imshow("y", flowy);
	imshow(name, qianjing);  ////显示前景图像
	//waitKey(3);
}






int main()
{

	int s = 1;

	unsigned long AAtime = 0, BBtime = 0;


	Mat GetImg, next, prvs;

	//gpu 变量
	cv::cuda::GpuMat prvs_gpu, next_gpu, flow_gpu;
	cv::cuda::GpuMat prvs_gpu_o, next_gpu_o;
	cv::cuda::GpuMat prvs_gpu_c, next_gpu_c;

	
	//VideoCapture stream1("20180718_093822.avi");     
	VideoCapture stream1(0);
	if (!(stream1.read(GetImg))) //get one frame form video
		return 0;



	//////////////////////////////////////////////////////////////////////////////////////////////
	//resize(GetImg, prvs, Size(GetImg.size().width/s, GetImg.size().height/s) );
	//cvtColor(prvs, prvs, CV_BGR2GRAY);
	//prvs_gpu.upload(prvs);
	//////////////////////////////////////////////////////////////////////////////////////////////
	//gpu upload, resize, color convert
	prvs_gpu_o.upload(GetImg);
	cv::cuda::resize(prvs_gpu_o, prvs_gpu_c, Size(GetImg.size().width / s, GetImg.size().height / s));
	cv::cuda::cvtColor(prvs_gpu_c, prvs_gpu, CV_BGR2GRAY);

	/////////////////////////////////////////////////////////////////////////////////////////////

	//dense optical flow
	//cv::cuda::FarnebackOpticalFlow fbOF;
	static Ptr<cuda::FarnebackOpticalFlow> fbOF = cuda::FarnebackOpticalFlow::create();
	int framenum = 0;

	while (true) {

		framenum++;

		if (!(stream1.read(GetImg))) //get one frame form video   
			break;

		///////////////////////////////////////////////////////////////////
		//resize(GetImg, next, Size(GetImg.size().width/s, GetImg.size().height/s) );
		//cvtColor(next, next, CV_BGR2GRAY);
		//next_gpu.upload(next);
		///////////////////////////////////////////////////////////////////
		//gpu upload, resize, color convert
		next_gpu_o.upload(GetImg);
		cv::cuda::resize(next_gpu_o, next_gpu_c, Size(GetImg.size().width / s, GetImg.size().height / s));
		cv::cuda::cvtColor(next_gpu_c, next_gpu, CV_BGR2GRAY);
		///////////////////////////////////////////////////////////////////

		AAtime = getTickCount();
		//dense optical flow
		fbOF->calc(prvs_gpu, next_gpu, flow_gpu);
		//fbOF(prvs_gpu, next_gpu, flow_x_gpu, flow_y_gpu);
		BBtime = getTickCount();
		float pt = (BBtime - AAtime) / getTickFrequency();
		float fpt = 1 / pt;
		printf("%.2lf / %.2lf \n", pt, fpt);

		//copy for vector flow drawing
	/*	Mat cflow;
		resize(GetImg, cflow, Size(GetImg.size().width / s, GetImg.size().height / s));*/
		//flow_x_gpu.download(flow_x);
		//flow_y_gpu.download(flow_y);
		//flow_gpu.download(cflow);

		//drawOptFlowMap_gpu(flow_x, flow_y, cflow, 10, CV_RGB(0, 255, 0));

		/*imshow("OpticalFlowFarneback", cflow);*/
		showFlow("OpticalFlowFarneback", flow_gpu, (GetImg.size().width / s),( GetImg.size().height / s),framenum);
		///////////////////////////////////////////////////////////////////
		//Display gpumat
		next_gpu.download(next);
		prvs_gpu.download(prvs);
		/*imshow("next", next);
		imshow("prvs", prvs);*/

		////prvs mat update
		prvs_gpu = next_gpu.clone();

		if (waitKey(5) >= 0)
			break;
	}
}