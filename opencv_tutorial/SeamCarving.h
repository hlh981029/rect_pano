#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
using namespace std;
using namespace cv;

enum Direction
{
	Top, Bottom, Left, Right, Horizontal, Vertical
};

struct BoundarySegment {
	int begin;
	int end;
	Direction direction;
};

class SeamCarving
{
public:
	SeamCarving(Mat& _image, Mat& _mask);
	BoundarySegment getLongestBoundary();
	void localWraping();
	void insertSeam();
	void calcCost();
	void showCost();
	int rows, cols;
	Mat& image;
	Mat& mask;
	Mat grayImage;
	Mat leftCost, upCost, rightCost, M, route;
	uchar* maskArray;
	float* grayImageArray;
	float* leftCostArray;
	float* upCostArray;
	float* rightCostArray;
	float* mArray;
	int* routeArray;
};

