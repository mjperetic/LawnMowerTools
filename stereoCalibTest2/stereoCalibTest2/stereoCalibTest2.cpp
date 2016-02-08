/* Software taken and modified from
Stereo Calibration demo
by Jay Rambhia

http://www.jayrambhia.com/blog/stereo-calibration/
http://www.jayrambhia.com/blog/stereo-calibration/

findChessboardCornersAndDraw adapted from
Abishek Upperwal's stereo vision calibration example
https://github.com/upperwal/opencv/blob/master/samples/cpp/stereo_calib.cpp

*/

// Standard includes
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>

// Project includes
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>

// The parameters for the user to set; these should be 
// converted to arguments if this routine is functionalized
#define HORIZ_CORNERS 9
#define VERTI_CORNERS 6
#define BOARD_CAPS 10
#define CAM_1_NUM 3
#define CAM_2_NUM 0

using namespace cv;
using namespace std;

int main()
{
	// Number of different board orientation pictures to take
	int numBoards = BOARD_CAPS;
	// Number of horizontal and vertical corners
	int board_w = HORIZ_CORNERS;
	int board_h = VERTI_CORNERS;

	// Package the board info
	Size board_sz = Size(board_w, board_h);
	// Determine the number of four square intersections
	int board_n = board_w*board_h;

	// Store points in vectors
	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > imagePoints1, imagePoints2;
	// Locations of corners in each image
	vector<Point2f> corners1, corners2;

	// Build a vector for all the corner points
	vector<Point3f> obj;
	for (int j = 0; j<board_n; j++)
	{
		obj.push_back(Point3f(j / board_w, j%board_w, 0.0f));
	}

	// Set up the capture images and their grayscale conversions
	Mat img1, img2, gray1, gray2;
	VideoCapture cap1(CAM_1_NUM);
	VideoCapture cap2(CAM_2_NUM);

	// Count the number of boards which found all the corners
	int success = 0, k = 0;
	// Flags for if all the corners were found for each image
	bool found1 = false;
	bool found2 = false;

	// Keep checking the image until shots of all boards have been acquired
	while (success < numBoards)
	{
		// Grab input from the user
		k = waitKey(10);

		// Keep refreshing the image until the user enters a character
		while (k == -1)
		{
			// Grab the image and convert to grayscale
			cap1 >> img1;
			cap2 >> img2;
			//resize(img1, img1, Size(320, 280));
			//resize(img2, img2, Size(320, 280));
			cvtColor(img1, gray1, CV_BGR2GRAY);
			cvtColor(img2, gray2, CV_BGR2GRAY);

			// Check each image for all the corners (true if all corners were found)
			found1 = findChessboardCorners(img1, board_sz, corners1, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
			found2 = findChessboardCorners(img2, board_sz, corners2, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
			//found1 = findChessboardCorners(img1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
			//found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

			if (found1)
			{
				// Refine the corners to a subpixel resolution
				cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.01));
				// Draw connected colored corners
				drawChessboardCorners(img1, board_sz, corners1, found1);
			}

			if (found2)
			{
				// Refine the corners to a subpixel resolution
				cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.01));
				// Draw connected colored corners
				drawChessboardCorners(img2, board_sz, corners2, found2);
			}

			// Display the two images (with the corners selected)
			imshow("Stereo Left", img1);
			imshow("Stereo Right", img2);

			// Update the inputted key
			k = waitKey(10);
		}

		if (k == 27)
		{
			// End now if the user entered ESC
			break;
		}
		// Hold the frame if the user pressed something other than ESC
		if (found1 != 0 && found2 != 0)
		{
			// Wait for the user to confirm/deconfirm the picture
			k = waitKey(0);

			// If the user pressed space, accept the corners
			if (k == ' ')
			{
				imagePoints1.push_back(corners1);
				imagePoints2.push_back(corners2);
				object_points.push_back(obj);
				printf("Corners stored\n");
				success++;

				// Quit the loop if we just had the max number of successes
				if (success >= numBoards)
				{
					break;
				}
			}
			// If the user entered a different character, go back to looking for frames
		}
	}
	
	// Close the pictures and the camera feeds
	destroyAllWindows();
	printf("Starting Calibration\n");

	// Determine the camera matrix for both cameras
	//Mat CM1 = Mat(3, 3, CV_64FC1);
	//Mat CM2 = Mat(3, 3, CV_64FC1);
	Mat CM1 = initCameraMatrix2D(object_points, imagePoints1, img1.size(), 0);
	Mat CM2 = initCameraMatrix2D(object_points, imagePoints2, img2.size(), 0);

	// Determine the distortion matrices for both camers
	Mat D1, D2;
	// Determine the rotation, translation, essential, and fundamental matrices
	Mat R, T, E, F;
	// Determine the termination criteria for the calibration algorithm 
	// (not gonna have perfect eigenvalues, so know when to stop)
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6);


	// Calibrate the cams!
	stereoCalibrate(object_points, imagePoints1, imagePoints2,
		CM1, D1, CM2, D2, img1.size(), R, T, E, F,
		CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST, criteria);


	// Original stereoCalibrate code -- pop up said criteria and flags were out of order, strangly
	// (everything online says flags and criteria are this way, though)
	//stereoCalibrate(object_points, imagePoints1, imagePoints2,
	//	CM1, D1, CM2, D2, img1.size(), R, T, E, F, criteria,
	//	CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST);

	// Store these params in a yml file
	FileStorage fs1("C:/users/mjper/Documents/stereoCalibParams.yml", FileStorage::WRITE);
	fs1 << "CM1" << CM1;
	fs1 << "CM2" << CM2;
	fs1 << "D1" << D1;
	fs1 << "D2" << D2;
	fs1 << "R" << R;
	fs1 << "T" << T;
	fs1 << "E" << E;
	fs1 << "F" << F;

	printf("Done Calibration\n");

	printf("Starting Rectification\n");

	// Determine transformation to rectify cameras
	// R1/R2 - rectification transform (rotation matrix to rectified space)
	// T1/T2 - projection transfrom (camera projection to rectified space)
	// Q - disparity to depth matrix (need to find out how this works, like units)
	Mat R1, R2, P1, P2, Q;
	stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
	fs1 << "R1" << R1;
	fs1 << "R2" << R2;
	fs1 << "P1" << P1;
	fs1 << "P2" << P2;
	fs1 << "Q" << Q;

	printf("Done Rectification\n");

	printf("Applying Undistort\n");

	// (not sure why the remaps have separate matrices for x and y
	Mat map1x, map1y, map2x, map2y;
	Mat imgU1, imgU2;

	// Undistort the images!
	initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
	initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);

	printf("Undistort complete\n");

	// Show * * L I V E * * A C T I O N * * video feed using the rectified video params
	while (1)
	{
		// Grab new images from the camera
		cap1 >> img1;
		cap2 >> img2;

		// Remap with the new params
		remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

		// Display them
		imshow("Stereo Left", imgU1);
		imshow("Stereo Right", imgU2);

		// Wait 5ms and check to see if the user hit ESC to end the feed
		k = waitKey(5);

		if (k == 27)
		{
			break;
		}
	}

	// Terminating the program, release the holds on the cameras
	cap1.release();
	cap2.release();

	return(0);
}