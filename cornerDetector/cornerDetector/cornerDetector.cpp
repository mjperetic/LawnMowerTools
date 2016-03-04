// cornerDetector.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Camera device specifications
#define CAM_L_NUM 3
#define CAM_R_NUM 0
#define CAM_WIDTH_PIX 640
#define CAM_HEIGHT_PIX 480

// Code setting defines
#define OVERLAP_INTERIOR_MODE 1

using namespace cv;
using namespace std;

Rect calculateOverlapBox(Mat map1x, Mat map1y, Mat map2x, Mat map2y);
Mat calculateHarrisCorners(Mat src);
Rect findMinRect(const Mat1b& src);
RotatedRect largestRectInNonConvexPoly(const Mat src);

int thresh = 165;
int max_thresh = 255;

// Constants used in displaying the matched boxes -- note OpenCV does BGR colors
Scalar greenColor = Scalar(0.0, 255.0, 0.0); 
Scalar redColor = Scalar(0.0, 0.0, 255.0); 
Scalar blueColor = Scalar(255.0, 0.0, 0.0);
Scalar orangeColor = Scalar(0.0, 220.0, 255.0);

int main() {
	// Camera image variables
	Mat imgL, imgR, imgL_over, imgR_over;

	// Initialize the camera matrix for both cameras
	Mat CM1 = Mat(3, 3, CV_64FC1);
	Mat CM2 = Mat(3, 3, CV_64FC1);

	// Initialize the distortion matrices for both camers
	Mat D1, D2;

	// R1/R2 - rectification transform (rotation matrix to rectified space)
	// T1/T2 - projection transfrom (camera projection to rectified space)
	// F - fundamental matrix
	Mat R1, R2, P1, P2, F;

	// Grab the calibration params from the yml file
	FileStorage fs1("C:/Users/mjper/Documents/stereoCalibParams.yml", FileStorage::READ);
	fs1["CM1"] >> CM1;
	fs1["CM2"] >> CM2;
	fs1["D1"] >> D1;
	fs1["D2"] >> D2;
	fs1["R1"] >> R1;
	fs1["R2"] >> R2;
	fs1["P1"] >> P1;
	fs1["P2"] >> P2;
	fs1["F"] >> F;

	// Create remapping matrices
	Mat mapLx, mapLy, mapRx, mapRy;
	Mat imgL_remap, imgR_remap;

	// Initialize the key watcher to a harmless value until receiving an input from the user
	char lastKey = 0;

	// Set up the capture images and their grayscale conversions
	VideoCapture capL(CAM_L_NUM);
	VideoCapture capR(CAM_R_NUM);

	// Grab an image from VideoCapture
	capL >> imgL;
	capR >> imgR;
	waitKey(100);

	// Grab a few more frames to make sure something works
	capL >> imgL;
	capR >> imgR;
	waitKey(100);

	// Undistort the images!
	initUndistortRectifyMap(CM1, D1, R1, P1, imgL.size(), CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(CM2, D2, R2, P2, imgR.size(), CV_32FC1, mapRx, mapRy);

	// Get the region contained in both images
	Rect sharedBox = calculateOverlapBox(mapLx, mapLy, mapRx, mapRy);

	// Create the window with threshold controls
	namedWindow("Controls", CV_WINDOW_NORMAL);
	createTrackbar("Corner Thresh:", "Controls", &thresh, max_thresh);

	// Loop until the user kills it by hitting 'ESC'
	while (lastKey != 27) {
		if (lastKey != 'p' && lastKey != ' ') {
			// Update the images
			capL >> imgL;
			capR >> imgR;
			// Remap with the new params
			remap(imgL, imgL_remap, mapLx, mapLy, INTER_LINEAR, BORDER_CONSTANT, Scalar());
			//img1_remap.copyTo(img1Mask);
			remap(imgR, imgR_remap, mapRx, mapRy, INTER_LINEAR, BORDER_CONSTANT, Scalar());
			//img2_remap.copyTo(img2Mask);

			// Prepare the overlay images
			imgL_over = imgL_remap;
			imgR_over = imgR_remap;

			// Create the harris corners map for both images
			Mat imgL_harris, imgR_harris;

			imgL_harris = calculateHarrisCorners(imgL_remap(sharedBox));
			imgR_harris = calculateHarrisCorners(imgR_remap(sharedBox));

			/// Drawing a circle around corners
			for (int j = 0; j < imgL_harris.rows; j++)
			{
				for (int i = 0; i < imgL_harris.cols; i++)
				{
					if ((int)imgL_harris.at<uchar>(j, i) > thresh)
					{
						circle(imgL_over, Point(i + sharedBox.x, j + sharedBox.y), 5, greenColor, 1, 8, 0);
					}

					if ((int)imgR_harris.at<uchar>(j, i) > thresh)
					{
						circle(imgR_over, Point(i + sharedBox.x, j + sharedBox.y), 5, greenColor, 1, 8, 0);
					}
				}
			}

			// Display the results
			imshow("Stereo Left", imgL_over);
			imshow("Stereo Right", imgR_over);

			Mat overlaidImgs;
			addWeighted(imgL_over, 0.5, imgR_over, 0.5, 0, overlaidImgs);

			rectangle(overlaidImgs, sharedBox, Scalar(255.0, 255.0, 255.0), 1, 8, 0);
			imshow("Overlaid Images", overlaidImgs);
		}
		else {
			// Block here until a new input is received (so no updates!)
			lastKey = waitKey(0);
		}

		// Grab the input from the user
		lastKey = waitKey(10);
	}

	destroyAllWindows();
	return(0);
}

// http://stackoverflow.com/a/30418912/5008845
// http://stackoverflow.com/questions/32674256/how-to-adapt-or-resize-a-rectangle-inside-an-object-without-including-or-with-a/32682512#32682512
Rect findMinRect(const Mat1b& src)
{
	Mat1f W(src.rows, src.cols, float(0));
	Mat1f H(src.rows, src.cols, float(0));

	Rect maxRect(0, 0, 0, 0);
	float maxArea = 0.f;

	for (int r = 0; r < src.rows; ++r)
	{
		for (int c = 0; c < src.cols; ++c)
		{
			if (src(r, c) == 0)
			{
				H(r, c) = 1.f + ((r>0) ? H(r - 1, c) : 0);
				W(r, c) = 1.f + ((c>0) ? W(r, c - 1) : 0);
			}

			float minw = W(r, c);
			for (int h = 0; h < H(r, c); ++h)
			{
				minw = min(minw, W(r - h, c));
				float area = (h + 1) * minw;
				if (area > maxArea)
				{
					maxArea = area;
					maxRect = Rect(Point(c - minw + 1, r - h), Point(c + 1, r + 1));
				}
			}
		}
	}

	return maxRect;
}


Rect calculateOverlapBox(Mat map1x, Mat map1y, Mat map2x, Mat map2y) {

	Rect overlapBox;

	// Create the masks used for block matching
	// Generate a white region the shape of the camera by remapping a white picture
	Mat whiteImage;
	Mat white_remap_imgL, white_remap_imgR;
	Mat white_shared_visible;
	Mat img1Mask;
	Mat img2Mask;

	whiteImage = 255.0 * Mat::ones(CAM_HEIGHT_PIX, CAM_WIDTH_PIX, CV_8UC1);
	remap(whiteImage, white_remap_imgL, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
	remap(whiteImage, white_remap_imgR, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
	bitwise_and(white_remap_imgL, white_remap_imgR, white_shared_visible);

	if (OVERLAP_INTERIOR_MODE) {
		Mat white_shared_visible_inverted;
		bitwise_not(white_shared_visible, white_shared_visible_inverted);
		overlapBox = findMinRect((Mat1b)white_shared_visible_inverted);
		//rectangle(white_shared_visible_inverted, sharedInteriorBox, Scalar(255.0), 1, 8, 0);
		//imshow("Interior Shared", white_shared_visible_inverted);
	}
	else {

		Mat white_shared_gray, white_thresh_out;
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		//imshow("Shared Vision", white_shared_visible);
		//cvtColor(white_shared_visible, white_shared_gray, CV_BGR2GRAY);
		/// Detect edges using Threshold
		//threshold(white_shared_visible, white_thresh_out, 127, 255, CV_THRESH_BINARY);
		Canny(white_shared_visible, white_thresh_out, 0, 30, 3);
		/// Find contours
		findContours(white_thresh_out, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

		// Bounding box of the contour of the overlapping region of the two rectified images

		// Keep just the largest bounding box, incase others show up for some reason
		for (int i = 0; i < contours.size(); i++) {
			vector<Point> contours_poly;
			// Kill sharp edges by approximating the contour with a polynomial
			//approxPolyDP(Mat(contours[i]), contours_poly, 3, true);
			// Create a bounding rectangle around the object
			Rect boundRect;
			boundRect = boundingRect(Mat(contours[i]));

			if ((boundRect.height * boundRect.width) >(overlapBox.height * overlapBox.width)) {
				overlapBox = boundRect;
			}
		}
	}

	return overlapBox;
}

Mat calculateHarrisCorners(Mat src) {

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	return dst_norm_scaled;
}