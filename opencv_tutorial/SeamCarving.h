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
    void placeMesh();
    Mat& image;
    Mat& mask;
    Mat grayImage, expandMaskImage, expandGrayImage, expandImage, seamImage, meshImage;
    Mat M, route;
    Direction directionMask, direction;
    int rows, cols, maxLen;
    int meshRows, meshCols;
    Point** mesh;
    int begin, end, left, right, middle, upLeft, up, upRight, rowOffset, minCostIndex;
    double tempUpCost, tempLeftCost, tempRightCost, minCost;
    bool hasLeft, hasRight, hasUp;
    int** neighborIndexArray;
    float* expandGrayArray;
    double* mArray, * mRowArray, * mUpRowArray;
    int* routeArray, * routeRowArray, * routeUpRowArray;
    uchar* expandImageArray, * expandImageRowArray, * expandImageUpRowArray;
    uchar* expandMaskArray, * expandMaskRowArray, * expandMaskUpRowArray;
    uchar* maskArray;

    Mat displacementIndex;
    int* displacementIndexArray, * displacementIndexRowArray;

    //Mat leftCost, upCost, rightCost;
    //Mat tMask, tGrayImage, tImageIndexUsed, tDisplacementIndex;
    //Mat imageIndexUsed, displacementIndex;
    //float* leftCostArray;
    //float* upCostArray;
    //float* rightCostArray;
    //uchar* imageIndexUsedArray;
};

