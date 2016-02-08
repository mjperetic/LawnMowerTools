// harrisTest.cpp : Defines the entry point for the console application.
//
// Taken from http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html
//

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

// Camera device specifications
#define CAM_1_NUM 3
#define CAM_2_NUM 0
// Set CONTOUR_DEBUG to 1 if you want the results displayed in windows; set it 0 if not
#define CONTOUR_DEBUG_OUTPUT 1
#define BOX_BIDI_DISPLAY 1

using namespace cv;
using namespace std;

int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

struct disparityPair {
	Mat img1;
	Mat img2;
};

struct boxData {
	Mat boxImg;
	Mat boxImgGray;
	Rect boxRect;
	int matchIndex = -1;
	double matchScore = -1;
	Point matchOffset;
	bool templateIm = false;
};

struct matchData {
	boxData img1Data;
	boxData img2Data;
};

/// Function header
Mat featureContourDetector(Mat src_img, bool dispBoxes);
vector<Rect> featureBoxDetector(Mat src_img);
disparityPair disparityCalculator(Mat src_img_left, Mat src_img_right);

// Constants used in displaying the matched boxes -- no sense defining each function call
Scalar goodMatchColor = Scalar(0.0, 220.0, 0.0); // Green
Scalar noMatchColor = Scalar(0.0, 0.0, 220.0); // This is red because OpenCV does BGR colors
Mat whiteImage;
Mat tempImage;
Mat img1Mask;
Mat img2Mask;


/** @function main */
int main() {

	// Camera image variables
	Mat img1, img2;
	Mat img1Mask, img2Mask;
	// Feature detector output variables
	Mat img1_feat, img2_feat;
	// Overlay image variables
	Mat img1_over, img2_over;
	disparityPair pairedBoxes;

	// Initialize the camera matrix for both cameras
	Mat CM1 = Mat(3, 3, CV_64FC1);
	Mat CM2 = Mat(3, 3, CV_64FC1);

	// Initialize the distortion matrices for both camers
	Mat D1, D2;

	// R1/R2 - rectification transform (rotation matrix to rectified space)
	// T1/T2 - projection transfrom (camera projection to rectified space)
	Mat R1, R2, P1, P2;

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

	// Remapping matrices
	Mat map1x, map1y, map2x, map2y;
	Mat img1_remap, img2_remap;

	// Create the disparity image
	Mat disparityImRaw;
	Mat disparityImProc;

	// Initialize the key watcher to a harmless value until receiving an input from the user
	char lastKey = 0;

	// Create the window with threshold controls
	namedWindow("Threshold Control", CV_WINDOW_AUTOSIZE);
	createTrackbar(" Threshold:", "Threshold Control", &thresh, max_thresh);

	// Set up the capture images and their grayscale conversions
	VideoCapture cap1(CAM_1_NUM);
	VideoCapture cap2(CAM_2_NUM);

	// Grab an image from VideoCapture
	cap1 >> img1;
	cap2 >> img2;
	waitKey(100);

	// Grab a few more frames to make sure something works
	cap1 >> img1;
	cap2 >> img2;
	waitKey(100);

	// Undistort the images!
	initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
	initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);

	// Create the masks used for block matching
	// Generate a white region the shape of the camera by remapping a white picture
	whiteImage = 255.0 * Mat::ones(img1.size(), CV_8UC3);
	remap(whiteImage, tempImage, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
	cvtColor(tempImage, img1Mask, CV_BGR2GRAY);
	remap(whiteImage, tempImage, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
	cvtColor(tempImage, img2Mask, CV_BGR2GRAY);

	// Loop until the user kills it
	while (lastKey != 27) {
		if (lastKey != 'p') {
			// Update the images
			cap1 >> img1;
			cap2 >> img2;
			// Remap with the new params
			remap(img1, img1_remap, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
			//img1_remap.copyTo(img1Mask);
			remap(img2, img2_remap, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
			//img2_remap.copyTo(img2Mask);

			// Get boxes for this pair
			if (BOX_BIDI_DISPLAY) {
				pairedBoxes = disparityCalculator(img1_remap, img2_remap);

				// Overlay the detected features on the image
				img1_over = img1_remap + pairedBoxes.img1;
				img2_over = img2_remap + pairedBoxes.img2;
			}
			else {
				// Generate an image of the features for each image
				img1_feat = featureContourDetector(img1_remap, 1);
				img2_feat = featureContourDetector(img2_remap, 0);

				// Overlay the detected features on the image
				img1_over = img1_remap + img1_feat;
				img2_over = img2_remap + img2_feat;
			}

			// Display the results
			imshow("Stereo Left", img1_over);
			imshow("Stereo Right", img2_over);
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

/** @function thresh_callback */
Mat featureContourDetector(Mat src_img, bool dispBoxes)
{
	Mat src_img_gray, threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat concatBoxesPrev, concatBoxesCur;
	Mat blankIm(src_img.rows, 1, src_img.type(), Scalar(0, 0, 0));
	concatBoxesPrev = blankIm;
	Mat test;

	// Convert image to gray and blur it
	cvtColor(src_img, src_img_gray, CV_BGR2GRAY);
	blur(src_img_gray, src_img_gray, Size(3, 3));

	// Variables for determing feature color
	double feat_r = 0.0;
	double feat_g = 0.0;
	double feat_b = 0.0;

	/// Detect edges using Threshold
	threshold(src_img_gray, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	vector<Mat>boxIm(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		// Kill sharp edges by approximating the contour with a polynomial
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		// Create a bounding rectangle around the object
		boundRect[i] = boundingRect(Mat(contours_poly[i]));

		//add(src_img(boundRect[i]), blankIm, test);
		//test = src_img(boundRect[i]) + blankIm;
		//resize(src_img(boundRect[i]), test, Size(boundRect[i].width, src_img.rows), 1, 1, 1);

		//hconcat(concatBoxesPrev, test, concatBoxesCur);
		//concatBoxesPrev = concatBoxesCur;

		// If processing is an issue, bypass the approxPolyDP and just use the sharp-edged contour
		//boundRect[i] = boundingRect(Mat(contours[i]));

		// Create a bounding circle around the object
		//minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}

	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		// Unnecessary calculations to try to give boxes colors that reflect their properties
		// (the only purpose this serves is aesthetics)
		feat_r = (200 * (((double)boundRect[i].height * (double)boundRect[i].width) / ((double)src_img.size().width * (double)src_img.size().height))) + 55;
		feat_g = (200 * (center[i].x / src_img.size().width)) + 55;
		feat_b = (200 * (center[i].y / src_img.size().height)) + 55;
		//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Scalar color = Scalar(feat_r, feat_g, feat_b); 

		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	return drawing;
}

vector<Rect> featureBoxDetector(Mat src_img) {

	Mat src_img_gray, threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Convert image to gray and blur it
	cvtColor(src_img, src_img_gray, CV_BGR2GRAY);
	blur(src_img_gray, src_img_gray, Size(3, 3));

	/// Detect edges using Threshold
	threshold(src_img_gray, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		// Approximate an outline for the contour using a polynomial
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		// Create a bounding rectangle around the object
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	return boundRect;
}

disparityPair disparityCalculator(Mat src_img1, Mat src_img2) {
	// Get bounding box info for each image
	vector<Rect> img1Boxes = featureBoxDetector(src_img1);
	vector<Rect> img2Boxes = featureBoxDetector(src_img2);
	Mat score;
	boxData img1CurData, img2CurData;

	// Output images with distances and boxes overlaid
	Mat img1BoxesOverlay = Mat::zeros(src_img1.size(), CV_8UC3);
	Mat img2BoxesOverlay = Mat::zeros(src_img1.size(), CV_8UC3);

	// Locations for text overlay
	Point img1TextLoc;
	Point img2TextLoc;


	// Make sure there are actually boxes
	if (!(img1Boxes.empty()) && !(img2Boxes.empty())) {
		// Box data
		vector<boxData> img1Data(img1Boxes.size());
		vector<boxData> img2Data(img2Boxes.size());
		double minVal = 0.0, maxVal = 0.0;
		Point minLoc, maxLoc;
		Mat maskFrag;
		int img1Iter, img2Iter;

		// Iterate through each box for each image
		for (img1Iter = 0; img1Iter < img1Boxes.size(); img1Iter++) {
			// Save the rectangle parameters and the appropriate part of the image for image 1
			img1Data[img1Iter].boxImg = src_img1(img1Boxes[img1Iter]);
			img1Data[img1Iter].boxRect = img1Boxes[img1Iter];
			cvtColor(src_img1(img1Boxes[img1Iter]), img1Data[img1Iter].boxImgGray, CV_BGR2GRAY);

			// Save the image data 
			for (img2Iter = 0; img2Iter < img2Boxes.size(); img2Iter++) {
				// Save the rectangle parameters and the appropriate part of the image for image 2, but only the first time through
				if (img1Iter == 0) {
					img2Data[img2Iter].boxImg = src_img2(img2Boxes[img2Iter]);
					img2Data[img2Iter].boxRect = img2Boxes[img2Iter];
					cvtColor(src_img2(img2Boxes[img2Iter]), img2Data[img2Iter].boxImgGray, CV_BGR2GRAY);
				}

				// Make sure there are valid images
				if (img1Data[img1Iter].boxImg.empty())
				{
					cout << "Nothing in Image 1!!!!!" << endl;
				}

				if (img2Data[img2Iter].boxImg.empty())
				{
					cout << "Nothing in Image 2!!!!!" << endl;
				}

				// If image 1 is the wider image
				if (img1Boxes[img1Iter].width >= img2Boxes[img2Iter].width) {
					// Check to make sure it's also the taller image
					if (img1Boxes[img1Iter].height >= img2Boxes[img2Iter].height)
					{
						//imshow("image", img1Data[img1Iter].boxImgGray);
						//imshow("template", img2Data[img2Iter].boxImgGray);
						//imshow("mask", img1Mask(img1Data[img1Iter].boxRect));
						//waitKey(10);
						//maskFrag = img1Mask(img1Data[img1Iter].boxRect);
						//maskFrag = img2Mask(img2Data[img2Iter].boxRect);
						// Sweep the smaller image over the larger image until a match is found
						//matchTemplate(img1Data[img1Iter].boxImgGray, img2Data[img2Iter].boxImgGray, score, CV_TM_CCORR_NORMED, maskFrag);
						matchTemplate(img1Data[img1Iter].boxImgGray, img2Data[img2Iter].boxImgGray, score, CV_TM_CCORR_NORMED);
					}
					else
					{							
						// If image 2 (the template) is the taller image, shorten it to the height of image 1
						// The taller parts of the image are kept so that references to the top corner of the images are consistent
						Rect crop = Rect(0, 0, img2Boxes[img2Iter].width, img1Boxes[img1Iter].height);
							
						//imshow("image", img1Data[img1Iter].boxImgGray);
						//imshow("template", img2Data[img2Iter].boxImgGray(crop));
						//imshow("mask", img1Mask(img1Data[img1Iter].boxRect));
						//waitKey(10);
							
						//maskFrag = img1Mask(img1Data[img1Iter].boxRect);
						//maskFrag = img2Mask(img2Data[img2Iter].boxRect);
						//matchTemplate(img1Data[img1Iter].boxImgGray, img2Data[img2Iter].boxImgGray(crop), score, CV_TM_CCORR_NORMED, maskFrag);
						matchTemplate(img1Data[img1Iter].boxImgGray, img2Data[img2Iter].boxImgGray(crop), score, CV_TM_CCORR_NORMED);

					}

					// Find the best match and where it is in the image
					minMaxLoc(score, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

					// Check this set of scores; if they're better, store them
					if (maxVal >= img1Data[img1Iter].matchScore) {
						img1Data[img1Iter].matchScore = maxVal;
						img1Data[img1Iter].matchOffset = maxLoc;
						img1Data[img1Iter].matchIndex = img2Iter;
						img1Data[img1Iter].templateIm = false;
					}
					// Likewise, have image 2 save these scores if they're better
					if (maxVal >= img2Data[img2Iter].matchScore) {
						img2Data[img2Iter].matchScore = maxVal;
						img2Data[img2Iter].matchOffset = maxLoc;
						img2Data[img2Iter].matchIndex = img1Iter;
						img2Data[img2Iter].templateIm = true;
					}
				}
				// If image 2 is the wider image
				else {
					// Check to make sure it's also the taller image
					if (img2Boxes[img2Iter].height >= img1Boxes[img1Iter].height)
					{
						//imshow("image", img2Data[img2Iter].boxImgGray);
						//imshow("template", img1Data[img1Iter].boxImgGray);
						//imshow("mask", img2Mask(img1Data[img2Iter].boxRect));
						//waitKey(10);
						//maskFrag = img2Mask(img2Data[img2Iter].boxRect);
						//maskFrag = img1Mask(img1Data[img1Iter].boxRect);
						// Sweep the smaller image over the larger image until a match is found
						//matchTemplate(img2Data[img2Iter].boxImgGray, img1Data[img1Iter].boxImgGray, score, CV_TM_CCORR_NORMED, maskFrag);
						matchTemplate(img2Data[img2Iter].boxImgGray, img1Data[img1Iter].boxImgGray, score, CV_TM_CCORR_NORMED);
					}
					else
					{
						// If image 1 (the template) is the taller image, shorten it to the height of image 2
						// The taller parts of the image are kept so that references to the top corner of the images are consistent
						Rect crop = Rect(0, 0, img1Boxes[img1Iter].width, img2Boxes[img2Iter].height);

						//imshow("image", img2Data[img2Iter].boxImgGray);
						//imshow("template", img1Data[img1Iter].boxImgGray(crop));
						//imshow("mask", img2Mask(img1Data[img2Iter].boxRect));
						//waitKey(10);
						//maskFrag = img2Mask(img2Data[img2Iter].boxRect);
						//maskFrag = img1Mask(img1Data[img1Iter].boxRect);
						//matchTemplate(img2Data[img2Iter].boxImgGray, img1Data[img1Iter].boxImgGray(crop), score, CV_TM_CCORR_NORMED, maskFrag);
						matchTemplate(img2Data[img2Iter].boxImgGray, img1Data[img1Iter].boxImgGray(crop), score, CV_TM_CCORR_NORMED);
					}

					// Find the best match and where it is in the image
					minMaxLoc(score, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

					// Check this set of scores; if they're better, store them
					if (maxVal >= img1Data[img1Iter].matchScore) {
						img1Data[img1Iter].matchScore = maxVal;
						img1Data[img1Iter].matchOffset = maxLoc;
						img1Data[img1Iter].matchIndex = img2Iter;
						img1Data[img1Iter].templateIm = true;
					}
					// Likewise, have image 2 save these scores if they're better
					if (maxVal >= img2Data[img2Iter].matchScore) {
						img2Data[img2Iter].matchScore = maxVal;
						img2Data[img2Iter].matchOffset = maxLoc;
						img2Data[img2Iter].matchIndex = img1Iter;
						img2Data[img2Iter].templateIm = false;
					}
				}
			} // End template match loop for image 2
		} // End template match loop for image 1

		// Perform bidirectional matching to decide which pairs of boxes to keep
		// Set up the bidirectional match vectors
		vector<matchData> goodBiDiMatch;
		matchData curBoxes;
		// Keep track of bad matches, too, if the user wants to see the results
		vector<boxData> img1FailedBiDiMatch;
		vector<boxData> img2FailedBiDiMatch;

		// Iterate through all ROI's in the first image
		for (img1Iter = 0; img1Iter < img1Boxes.size(); img1Iter++) {
			// Compare the index for the best match in image 2 to the current index for the box in image 1 and save both if they match
			if (img1Iter == img2Data[img1Data[img1Iter].matchIndex].matchIndex) {
				curBoxes.img1Data = img1Data[img1Iter];
				curBoxes.img2Data = img2Data[img1Data[img1Iter].matchIndex];
				goodBiDiMatch.push_back(curBoxes);
			}
			else {
				img1FailedBiDiMatch.push_back(img1Data[img1Iter]);
			}
		}

		// Iterate through all the ROI's in the second image
		for (img2Iter = 0; img2Iter < img2Boxes.size(); img2Iter++) {
			// Save only the failed matches; we got all the bidirectional matches from the previous matches
			if (img2Iter != img1Data[img2Data[img2Iter].matchIndex].matchIndex) {
				img2FailedBiDiMatch.push_back(img2Data[img2Iter]);
			}
		}

		// Draw the good matches
		for (int i = 0; i < goodBiDiMatch.size(); i++) {
			rectangle(img1BoxesOverlay, goodBiDiMatch[i].img1Data.boxRect, goodMatchColor, 2, 8, 0);
			rectangle(img2BoxesOverlay, goodBiDiMatch[i].img2Data.boxRect, goodMatchColor, 2, 8, 0);
			/*
			// If the first image is the template, have the point for the text for image 2 be take of the offset into account
			if (img1Data[i].templateIm) {
				//img1TextLoc = Point(src_img1.size().width - (img1Data[i].boxRect.x + (0.5 * img1Data[i].boxRect.width)), src_img1.size().height - (img1Data[i].boxRect.y + (0.5 * img1Data[i].boxRect.height)));
				//img2TextLoc = Point(src_img2.size().width - (img2Data[i].boxRect.x + img2Data[i].matchOffset.x + (0.5 * img2Data[i].boxRect.width)), src_img2.size().height - (img2Data[i].boxRect.y + img2Data[i].matchOffset.y + (0.5 * img2Data[i].boxRect.height)));
				img1TextLoc = Point(src_img1.size().height - img1Data[i].boxRect.y, src_img1.size().width - img1Data[i].boxRect.x);
				img2TextLoc = Point(src_img2.size().height - img2Data[i].boxRect.y, src_img2.size().width - img2Data[i].boxRect.x);
			}
			// Otherwise, have the point for the text for image 1 take the offset into account
			else {
				//img1TextLoc = Point(src_img1.size().width - (img1Data[i].boxRect.x + img1Data[i].matchOffset.x + (0.5 * img1Data[i].boxRect.width)), src_img1.size().height - (img1Data[i].boxRect.y + img1Data[i].matchOffset.y + (0.5 * img1Data[i].boxRect.height)));
				//img2TextLoc = Point(src_img2.size().width - (img2Data[i].boxRect.x + (0.5 * img2Data[i].boxRect.width)), src_img2.size().height - (img2Data[i].boxRect.y + (0.5 * img2Data[i].boxRect.height)));
				img1TextLoc = Point(src_img1.size().height - img1Data[i].boxRect.y, src_img1.size().width - img1Data[i].boxRect.x);
				img2TextLoc = Point(src_img2.size().height - img2Data[i].boxRect.y, src_img2.size().width - img2Data[i].boxRect.x);
			}
			*/
			// Place identifying numbers on the boxes
			//putText(img1BoxesOverlay, to_string(i), img1TextLoc, CV_FONT_HERSHEY_PLAIN, 1, goodMatchColor, 1, 8);
			//putText(img2BoxesOverlay, to_string(i), img2TextLoc, CV_FONT_HERSHEY_PLAIN, 1, goodMatchColor, 1, 8);
		}
		// Draw the bad matches
		//for (int i = 0; i < img1FailedBiDiMatch.size(); i++) {
		//	rectangle(img1BoxesOverlay, img1FailedBiDiMatch[i].boxRect, noMatchColor, 2, 8, 0);
		//}
		//for (int i = 0; i < img2FailedBiDiMatch.size(); i++) {
		//	rectangle(img2BoxesOverlay, img2FailedBiDiMatch[i].boxRect, noMatchColor, 2, 8, 0);
		//}

	} // end box check iterator


	// Return stuff
	disparityPair result;
	result.img1 = img1BoxesOverlay;
	result.img2 = img2BoxesOverlay;

	return result;
}