#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>
#include "Utils.h"

using namespace std;
using namespace cv;

enum Direction
{
	None, Horizontal = 3, Vertical = 12, Top = 1, Bottom = 2, Left = 4, Right = 8
};

struct BoundarySegment {
	int begin;
	int end;
	Direction direction;
	BoundarySegment(int _begin = 0, int _end = 0, Direction _direction = Direction::None);
	void print();
};

class SeamCarving
{
public:
	SeamCarving(Mat& _image, Mat& _mask);
	BoundarySegment getLongestBoundary();
	void localWraping();
	void insertSeam(BoundarySegment boundarySegment);
	void calcCost(BoundarySegment boundarySegment);
	void showCost(BoundarySegment boundarySegment);
	int rows, cols, maxLen;
	Mat& image;
	Mat& mask;
	Mat grayImage;
	Mat expandMaskImage, expandGrayImage, expandImage;
	float* expandGrayArray;
	uchar* expandMaskArray;
	int** neighborIndexArray;
	Vec3b temp;
	//Mat tMask, tGrayImage, tImageIndexUsed, tDisplacementIndex;
	Mat /*leftCost, upCost, rightCost,*/ M, route;
	//Mat imageIndexUsed, displacementIndex;
	uchar* maskArray;
	//float* leftCostArray;
	//float* upCostArray;
	//float* rightCostArray;
	float* mArray;
	//uchar* imageIndexUsedArray;
	//int* displacementIndexArray;
	int* routeArray;
	Direction directionMask;
};

