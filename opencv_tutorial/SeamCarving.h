#pragma once
#include "Utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;

enum Direction
{
    None, Horizontal = 3, Vertical = 12, Top = 1, Bottom = 2, Left = 4, Right = 8
};

struct BoundarySegment {
    int begin;
    int end;
    Direction direction;
    BoundarySegment(int _begin = 0, int _end = 0, Direction _direction = Direction::None);
    int length() {
        return end - begin;
    }
    void print();
};

class SeamCarving
{
public:
    SeamCarving(cv::Mat& _image, cv::Mat& _mask);
    BoundarySegment getLongestBoundary();
    void localWarping();
    void insertSeam(BoundarySegment boundarySegment);
    void calcCost(BoundarySegment boundarySegment);
    void showCost(BoundarySegment boundarySegment);
    void placeMesh();
    cv::Mat& image;
    cv::Mat& mask;
    cv::Mat grayImage, expandMaskImage, expandGrayImage, expandImage, seamImage;
    cv::Mat M, route;
    Direction directionMask, direction;
    int rows, cols, maxLen;
    int meshRows, meshCols;
    cv::Point** mesh;
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
    cv::Mat displacementIndex;
    int* displacementIndexArray, * displacementIndexRowArray;
};

