// camera.cpp
#include"stdafx.h"
#include "opencv2/objdetect.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/imgproc.hpp"  
#include <iostream>  
#include <stdio.h>  
using namespace std;
using namespace cv;
void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, bool tryflip);
string cascadeName = "cascade.xml";
int main(int argc, const char** argv)
{
	Mat image;
	bool tryflip = false;//no flip of face first
	CascadeClassifier cascade, nestedCascade;//give the classifier 
	double scale = 1;//give the scale
	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}
	if (argc == 1)
	{
		VideoCapture cap(0);      
		if (!cap.isOpened())
		{
			return -1;
		}
		// catch frame in cycle  
		bool stop = false;
		while (!stop)
		{
			cap >> image;
			cvNamedWindow("result", 1);
			if (!image.empty())
			{
				detectAndDraw(image, cascade, scale, tryflip);
			}
			/*if (waitKey(30) >= 0)
				stop = true;*/
		}
	}
	else
	{
		image = imread(argv[1], 1);
		cout << "In image read" << endl;
		cvNamedWindow("result", 1);
		if (!image.empty())
		{
			detectAndDraw(image, cascade, scale, tryflip);
			waitKey(0);
		}
	}
	//cvDestroyWindow("result");
	system("Pause");
	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, bool tryflip)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces, faces2;//face1 before flip,face2 after flip
	const static Scalar colors[] = { CV_RGB(0,0,255),CV_RGB(0,128,255),CV_RGB(0,255,255),CV_RGB(0,255,0),
		CV_RGB(255,128,0),CV_RGB(255,255,0),CV_RGB(255,0,0),CV_RGB(255,0,255) };//give the colors to the circle  
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	cvtColor(img, gray, COLOR_BGR2GRAY);//change to gray 
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//Bilinear interpolation 
	equalizeHist(smallImg, smallImg);//Histogram equalization

	t = (double)cvGetTickCount();//get time now 
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CASCADE_FIND_BIGGEST_OBJECT  
		//|CASCADE_DO_ROUGH_SEARCH  
		| CASCADE_SCALE_IMAGE
		,
		Size(30, 30));
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT  
			//|CASCADE_DO_ROUGH_SEARCH  
			| CASCADE_SCALE_IMAGE
			,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)cvGetTickCount() - t;//get the time used 
	printf("detection time = %g ms\nthe number of faces = %d\n", t / ((double)cvGetTickFrequency()*1000.), faces.size());
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)//draw circle 
	{
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r->width / r->height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r->x + r->width*0.5)*scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = cvRound((r->width + r->height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
				cvPoint(cvRound((r->x + r->width - 1)*scale), cvRound((r->y + r->height - 1)*scale)),
				color, 3, 8, 0);
		smallImgROI = smallImg(*r);
	}
	cv::imshow("result", img);
	//imwrite("Image.jpg", img);
	waitKey(50);
}