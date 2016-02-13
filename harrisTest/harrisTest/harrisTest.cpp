// harrisTest.cpp : Defines the entry point for the console application.
//
// Taken from http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html
//

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
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
// Only take child contours
#define JERRY_MODE 1

using namespace cv;
using namespace std;

int thresh = 100;
int max_thresh = 255;
int patchThresh = 10;
int max_patchThresh = 50;
int blurThresh = 8;
int max_blurThresh = 50;
int epsilonThresh = 6;
int max_epsilonThresh = 20;
int thresholdMode = 0;
int max_thresholdMode = 4;
RNG rng(12345);

struct disparityPair {
	Mat img1;
	Mat img2;
};

// Class to simplify calculation y for a given x on an epipolar line
class epiLine {
	float a, b, c;
public:
	epiLine(float, float, float);
	epiLine();
	int yVal(int x) {
		if (b != 0) {
			return (int)(-(a * x + c) / b);
		}
		else {
			return -1;
		}
	}
};

// Constructor for epipolar lines
epiLine::epiLine(float paramA, float paramB, float paramC) {
	a = paramA;
	b = paramB;
	c = paramC;
}
epiLine::epiLine() {
	// Guarantee that the line is garbage by setting everything to zero
	a = 0;
	b = 0;
	c = 0;
}

struct boxData {
	Mat boxImg;
	Mat boxImgGray;
	Rect boxRect;
	int matchIndex = -1;
	double matchScore = -1;
	Point matchOffset;
	bool templateIm = false;
	epiLine topEpiBound;
	epiLine botEpiBound;
	boxData(epiLine,epiLine);
	boxData();
};

boxData::boxData(epiLine epi1, epiLine epi2) {
	topEpiBound = epiLine(epi1);
	botEpiBound = epiLine(epi2);
}

boxData::boxData() {
	topEpiBound = epiLine();
	botEpiBound = epiLine();
}

struct matchData {
	boxData img1Data;
	boxData img2Data;
	matchData(boxData, boxData);
};

matchData::matchData(boxData img1, boxData img2) {
	img1Data = img1;
	img2Data = img2;
}

//matchData::matchData(boxData img1, boxData img2) {
//	img1Data = img1;
//	img2Data = img2;
//}


/// Function header
Mat featureContourDetector(Mat src_img, bool dispBoxes, Mat F);
vector<Rect> featureBoxDetector(Mat src_img);
disparityPair disparityCalculator(Mat src_img_left, Mat src_img_right, Mat F);

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
	// Epipolar lines
	vector<Vec3f> epiLinesImg1in2;
	vector<Vec3f> epiLinesImg2in1;

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
	Mat map1x, map1y, map2x, map2y;
	Mat img1_remap, img2_remap;
	

	// Initialize the key watcher to a harmless value until receiving an input from the user
	char lastKey = 0;

	// Create the window with threshold controls
	namedWindow("Controls", CV_WINDOW_NORMAL);
	createTrackbar("Contour Thresh:", "Controls", &thresh, max_thresh);
	createTrackbar("Box Thresh:", "Controls", &patchThresh, max_patchThresh);
	createTrackbar("Blur Thresh:", "Controls", &blurThresh, max_blurThresh);
	createTrackbar("Epsilon Thresh:", "Controls", &epsilonThresh, max_epsilonThresh);
	createTrackbar("Threshold Mode:", "Controls", &thresholdMode, max_thresholdMode);

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
				pairedBoxes = disparityCalculator(img1_remap, img2_remap, F);

				// Overlay the detected features on the image
				img1_over = img1_remap + pairedBoxes.img1;
				img2_over = img2_remap + pairedBoxes.img2;
			}
			else {
				// Generate an image of the features for each image
				img1_feat = featureContourDetector(img1_remap, 1, F);
				img2_feat = featureContourDetector(img2_remap, 0, F);

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
Mat featureContourDetector(Mat src_img, bool dispBoxes, Mat F)
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
	GaussianBlur(src_img_gray, src_img_gray, Size(2 * blurThresh + 1, 2 * blurThresh + 1), epsilonThresh * 0.5, epsilonThresh * 0.5);

	// Variables for determing feature color
	double feat_r = 0.0;
	double feat_g = 0.0;
	double feat_b = 0.0;

	/// Detect edges using Threshold
	threshold(src_img_gray, threshold_output, thresh, 255, thresholdMode);
	/// Find contours
	// Hierarchy returns information about contour i in the form of a 2D array
	// hierarchy[i] = {Next, Previous, Child, Parent}
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	vector<Mat>boxIm(contours.size());
	
	vector<vector<Point>> contours_poly_kids;	
	vector<Rect> boundRect_kids;
	vector<vector<Point>> contours_poly_parents;
	vector<Rect> boundRect_parents;

	for (int i = 0; i < contours.size(); i++)
	{
		if (JERRY_MODE) {
			// Separate the child rectangles from the parents
			if (hierarchy[i][2] == -1) {
				vector<Point> contours_poly_single;
				Rect boundRect_single;

				// Kill sharp edges by approximating the child contour 
				approxPolyDP(Mat(contours[i]), contours_poly_single, 3, true);

				// Create bounding boxes around the child object
				boundRect_single = boundingRect(Mat(contours_poly_single));

				// Don't save any "contours" that are 1 pixel wide or tall
				if (boundRect_single.width > patchThresh && boundRect_single.height > patchThresh) {
					contours_poly_kids.push_back(contours_poly_single);
					boundRect_kids.push_back(boundRect_single);
				}
			}
			else {
				vector<Point> contours_poly_single;
				Rect boundRect_single;

				// Kill sharp edges by approximating the child contour 
				approxPolyDP(Mat(contours[i]), contours_poly_single, 3, true);
				contours_poly_parents.push_back(contours_poly_single);

				// Create bounding boxes around the child object
				boundRect_single = boundingRect(Mat(contours_poly_single));
				boundRect_parents.push_back(boundRect_single);
			}
		}
		else {
			// Kill sharp edges by approximating the contour with a polynomial
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			// Create a bounding rectangle around the object
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}

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
	if (JERRY_MODE) {
		for (int i = 0; i < contours_poly_kids.size(); i++) {
			drawContours(drawing, contours_poly_kids, i, goodMatchColor, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(drawing, boundRect_kids[i].tl(), boundRect_kids[i].br(), goodMatchColor, 2, 8, 0);
			// Place identifying numbers on the boxes
			Point boxTextLoc = Point(boundRect_kids[i].x, boundRect_kids[i].y);
			putText(drawing, to_string(i), boxTextLoc, CV_FONT_HERSHEY_PLAIN, 1, goodMatchColor, 1, 8);
		}

		for (int i = 0; i < contours_poly_parents.size(); i++) {
			drawContours(drawing, contours_poly_parents, i, noMatchColor, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(drawing, boundRect_parents[i].tl(), boundRect_parents[i].br(), noMatchColor, 2, 8, 0);
		}
	}
	else {
		for (int i = 0; i < contours.size(); i++)
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
	}

	if (JERRY_MODE) {
		for (int i = 0; i < boundRect_kids.size(); i++) {
			vector<Vec3f> epiLines;
			vector<Point> boxEdgePoints;
			boxEdgePoints.push_back(Point(boundRect_kids[i].x + (0.5 * boundRect_kids[i].width), boundRect_kids[i].y));
			boxEdgePoints.push_back(Point(boundRect_kids[i].x + (0.5 * boundRect_kids[i].width), boundRect_kids[i].y + boundRect_kids[i].height));
			
			computeCorrespondEpilines(boxEdgePoints, 1, F, epiLines);
			if (dispBoxes) {
				for (int j = 0; j < epiLines.size(); j++) {
					Point startPt = Point(0, -epiLines[j][2] / epiLines[j][1]);
					Point endPt = Point(src_img.size().width, -(epiLines[j][0] * src_img.size().width + epiLines[j][2]) / epiLines[j][1]);
					line(drawing, startPt, endPt, goodMatchColor, 1, 8, 0);
				}
			}
		}
		
	}




	return drawing;
}

vector<Rect> featureBoxDetector(Mat src_img) {

	Mat src_img_gray, threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Convert image to gray and blur it
	cvtColor(src_img, src_img_gray, CV_BGR2GRAY);
	GaussianBlur(src_img_gray, src_img_gray, Size(2 * blurThresh + 1, 2 * blurThresh + 1), epsilonThresh * 0.5, epsilonThresh * 0.5);

	/// Detect edges using Threshold
	threshold(src_img_gray, threshold_output, thresh, 255, thresholdMode);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());

	// Approximation stuff for keeping the lowest level contours
	vector<vector<Point>> contours_poly_kids;
	vector<Rect> boundRect_kids;

	vector<Rect> boundRectOutput;

	for (int i = 0; i < contours.size(); i++) {
		if (JERRY_MODE) {
			// Separate the child rectangles from the parents
			if (hierarchy[i][2] == -1) {
				// Since the total number of kids are unknown at this point, contours have to be evaluated separately
				// then added to the vector via push_back, so each contour needs its own variable to be added to the vector.
				vector<Point> contours_poly_single;
				Rect boundRect_single;

				// Kill sharp edges by approximating the child contour 
				approxPolyDP(Mat(contours[i]), contours_poly_single, 3, true);

				// Create bounding boxes around the child object
				boundRect_single = boundingRect(Mat(contours_poly_single));

				// Don't save any "contours" that are 1 pixel wide or tall
				if (boundRect_single.width > patchThresh && boundRect_single.height > patchThresh) {
					contours_poly_kids.push_back(contours_poly_single);
					boundRect_kids.push_back(boundRect_single);
				}
			}
		}
		else {
			// Kill sharp edges by approximating the contour with a polynomial
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			// Create a bounding rectangle around the object
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}
	}

	// Return either just the lowest level contours or all the contours depending on the operation mode
	if (JERRY_MODE)	{
		return boundRect_kids;
	}
	else {
		return boundRect;
	}
}

disparityPair disparityCalculator(Mat src_img1, Mat src_img2, Mat F) {
	// Get bounding box info for each image
	vector<Rect> img1Boxes = featureBoxDetector(src_img1);
	vector<Rect> img2Boxes = featureBoxDetector(src_img2);
	Mat score;

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

			// Calculate epipolar lines from the middle top and middle bottom of the current box
			vector<Vec3f> curEpiLines;
			vector<Point> boxEdgePoints;
			boxEdgePoints.push_back(Point(img1Data[img1Iter].boxRect.x, img1Data[img1Iter].boxRect.y - (0.5 * img1Data[img1Iter].boxRect.height)));
			boxEdgePoints.push_back(Point(img1Data[img1Iter].boxRect.x, img1Data[img1Iter].boxRect.y + (1.5 * img1Data[img1Iter].boxRect.height)));
			computeCorrespondEpilines(boxEdgePoints, 1, F, curEpiLines);

			// Save the epipolar lines for this box
			img1Data[img1Iter].topEpiBound = epiLine(curEpiLines[0][0], curEpiLines[0][1], curEpiLines[0][2]);
			img1Data[img1Iter].botEpiBound = epiLine(curEpiLines[1][0], curEpiLines[1][1], curEpiLines[1][2]);

			// Save the image data for image 2 if this is the first loop through the data
			for (img2Iter = 0; img2Iter < img2Boxes.size(); img2Iter++) {
				// Save the rectangle parameters and the appropriate part of the image for image 2, but only the first time through
				if (img1Iter == 0) {
					img2Data[img2Iter].boxImg = src_img2(img2Boxes[img2Iter]);
					img2Data[img2Iter].boxRect = img2Boxes[img2Iter];
					cvtColor(src_img2(img2Boxes[img2Iter]), img2Data[img2Iter].boxImgGray, CV_BGR2GRAY);
				}

				// Match the box only if it's within the epipolar boundaries

				// Note that the y value for a top epipolar line will be a smaller number than the y value for a bot epipolar line
				// since y references the column number of the image. Thus, to be within a line-bounded region, a box must have
				// its top left corner be a larger number than the corresponding y on the top line and its bottom left corner
				// be a smaller number than the corresponding y on the bot line

				if (img2Data[img2Iter].boxRect.y > img1Data[img1Iter].topEpiBound.yVal(img2Data[img2Iter].boxRect.x) && (img2Data[img2Iter].boxRect.y + img2Data[img2Iter].boxRect.height) < img1Data[img1Iter].botEpiBound.yVal(img2Data[img2Iter].boxRect.x)) {
					// If image 1 is the wider image
					if (img1Boxes[img1Iter].width >= img2Boxes[img2Iter].width) {
						// Check to make sure it's also the taller image
						if (img1Boxes[img1Iter].height >= img2Boxes[img2Iter].height)
						{
							matchTemplate(img1Data[img1Iter].boxImgGray, img2Data[img2Iter].boxImgGray, score, CV_TM_CCORR_NORMED);
						}
						else
						{
							// If image 2 (the template) is the taller image, shorten it to the height of image 1
							// The taller parts of the image are kept so that references to the top corner of the images are consistent
							Rect crop = Rect(0, 0, img2Boxes[img2Iter].width, img1Boxes[img1Iter].height);
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
						if (img2Boxes[img2Iter].height >= img1Boxes[img1Iter].height) {
							matchTemplate(img2Data[img2Iter].boxImgGray, img1Data[img1Iter].boxImgGray, score, CV_TM_CCORR_NORMED);
						}
						else {
							// If image 1 (the template) is the taller image, shorten it to the height of image 2
							// The taller parts of the image are kept so that references to the top corner of the images are consistent
							Rect crop = Rect(0, 0, img1Boxes[img1Iter].width, img2Boxes[img2Iter].height);
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
				} // End epipolar lines boundary check
			} // End loop for boxes in image 2
		} // End loop for boxes in image 1


		// Perform bidirectional matching to decide which pairs of boxes to keep
		// Set up the bidirectional match vectors
		vector<matchData> goodBiDiMatch;
		//matchData curBoxes;
		// Keep track of bad matches, too, if the user wants to see the results
		vector<boxData> img1FailedBiDiMatch;
		vector<boxData> img2FailedBiDiMatch;

		// Iterate through all ROI's in the first image
		for (img1Iter = 0; img1Iter < img1Boxes.size(); img1Iter++) {
			// Check that this box has tried to match to something (may never match if no boxes fall within the epi boarder)
			if (img1Data[img1Iter].matchIndex != -1) {
				// Compare the index for the best match in image 2 to the current index for the box in image 1 and save both if they match
				if (img1Iter == img2Data[img1Data[img1Iter].matchIndex].matchIndex) {
					goodBiDiMatch.push_back(matchData(img1Data[img1Iter], img2Data[img1Data[img1Iter].matchIndex]));

				}
				else {
					img1FailedBiDiMatch.push_back(img1Data[img1Iter]);
				}
			}
		}

		// Iterate through all the ROI's in the second image
		for (img2Iter = 0; img2Iter < img2Boxes.size(); img2Iter++) {
			// Check that this box has tried to match to something (may never match if no boxes fall within the epi boarder)
			if (img2Data[img2Iter].matchIndex != -1) {
				// Save only the failed matches; we got all the bidirectional matches from the previous matches
				if (img2Iter != img1Data[img2Data[img2Iter].matchIndex].matchIndex) {
					img2FailedBiDiMatch.push_back(img2Data[img2Iter]);
				}
			}
		}

		// Draw the good matches
		for (int i = 0; i < goodBiDiMatch.size(); i++) {
			// Draw boxes
			rectangle(img1BoxesOverlay, goodBiDiMatch[i].img1Data.boxRect, goodMatchColor, 2, 8, 0);
			rectangle(img2BoxesOverlay, goodBiDiMatch[i].img2Data.boxRect, goodMatchColor, 2, 8, 0);

			// Number boxes
			putText(img1BoxesOverlay, to_string(i), Point(goodBiDiMatch[i].img1Data.boxRect.x, goodBiDiMatch[i].img1Data.boxRect.y), CV_FONT_HERSHEY_PLAIN, 1, goodMatchColor, 1, 8);
			putText(img2BoxesOverlay, to_string(i), Point(goodBiDiMatch[i].img2Data.boxRect.x, goodBiDiMatch[i].img2Data.boxRect.y), CV_FONT_HERSHEY_PLAIN, 1, goodMatchColor, 1, 8);
	
			// Draw epipolar lines (just for image 1; image 2 doesn't calculate epipolar lines)
			line(img2BoxesOverlay, Point(0, goodBiDiMatch[i].img1Data.topEpiBound.yVal(0)), Point(src_img1.size().width, goodBiDiMatch[i].img1Data.topEpiBound.yVal(src_img1.size().width)), goodMatchColor, 1, 8, 0);
			line(img2BoxesOverlay, Point(0, goodBiDiMatch[i].img1Data.botEpiBound.yVal(0)), Point(src_img1.size().width, goodBiDiMatch[i].img1Data.botEpiBound.yVal(src_img1.size().width)), goodMatchColor, 1, 8, 0);
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