#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "Utils.h"

extern "C"
{
#include "lsd.h";
}

using namespace std;
using namespace Eigen;

struct BilinearWeight {
    double u, v;
    BilinearWeight(double _u, double _v) :u(_u), v(_v) {}
    BilinearWeight(const BilinearWeight& p) :u(p.u), v(p.v) {}
    BilinearWeight() :u(0), v(0) {}
};

struct Coordinate {
    double row, col;
    Coordinate(double _col, double _row) :row(_row), col(_col) {}
    Coordinate(const Coordinate& p) :row(p.row), col(p.col) {}
    Coordinate() :row(0), col(0) {}
    const cv::Point toPoint() {
        return cv::Point(col, row);
    }
    friend const Coordinate operator-(const Coordinate& p1, const Coordinate& p2) {
        return Coordinate(p1.col - p2.col, p1.row - p2.row);
    }
    friend const Coordinate operator+(const Coordinate& p1, const Coordinate& p2) {
        return Coordinate(p1.col + p2.col, p1.row + p2.row);
    }
    friend ostream& operator<<(ostream& stream, const Coordinate& p) {
        stream << "(" << p.col << "," << p.row << ")";
        return stream;
    }
};

struct LineSegment {
    double col1, col2, row1, row2;
    LineSegment(double _col1, double _row1, double _col2, double _row2)
        :row1(_row1), col1(_col1), row2(_row2), col2(_col2) {}
    LineSegment(Coordinate &lp1, Coordinate &lp2)
        :row1(lp1.row), col1(lp1.col), row2(lp2.row), col2(lp2.col) {}
    LineSegment(const LineSegment &ls)
        :row1(ls.row1), col1(ls.col1), row2(ls.row2), col2(ls.col2) {}
    LineSegment()
        :row1(0), col1(0), row2(0), col2(0) {}
    const cv::Point toPoint1() {
        return cv::Point(col1, row1);
    }
    const cv::Point toPoint2() {
        return cv::Point(col2, row2);
    }
    const Coordinate toCord1() {
        return Coordinate(col1, row1);
    }
    const Coordinate toCord2() {
        return Coordinate(col2, row2);
    }
    const Vector2d toVector2D() {
        return Vector2d(col1 - col2, row1 - row2);
    }
    const double length() {
        return sqrt((col1 - col2) * (col1 - col2) + (row1 - row2) * (row1 - row2));
    }
};

class GlobalWraping
{
public:
    GlobalWraping(cv::Mat& _image, cv::Mat& _mask, cv::Point** mesh, int _meshRows, int _meshCols);
    void calcMeshToVertex();
    void calcMeshShapeEnergy();
    void calcBoundaryEnergy();
    void calcMeshLineEnergy();
    void detectLineSegment();
    void cutLineSegment(vector<LineSegment>& lineSegments);
    void calcBilinearWeight();
    void calcRadianBin();
    void updateV();
    bool inQuad(Coordinate point, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight);
    vector<Coordinate> getIntersectionWithQuad(LineSegment line, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight);
    bool lineIntersectLine(LineSegment line, double slope, double intersect, bool vertical, Coordinate& intersectPoint);
    BilinearWeight getBilinearWeight(Coordinate point, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight);
    void drawMesh(Coordinate** mesh);
    void calcCost(Coordinate** mesh);
    void calcLineCost(Coordinate** mesh);
    void updateTheta();
    void test(Coordinate** mesh, string str);

    cv::Mat& image;
    cv::Mat& mask;
    int cols, rows, meshRows, meshCols;
    int meshLineNumber;
    Coordinate** meshVertex;
    Coordinate** newMeshVertex;
    int*** meshArray;
    vector<LineSegment>** meshLineSegment;
    vector<MatrixXd>** meshLineBilinearWeight;
    vector<pair<MatrixXd, MatrixXd>>** meshLinePointBilinearWeight;
    vector<pair<double, int>>** meshLineRadianBin;
    vector<double>** meshLineRotation;
    VectorXd boundaryY;
    SparseMatrix<double, RowMajor> meshToVertex;
    SparseMatrix<double, RowMajor> meshShapeEnergy;
    SparseMatrix<double, RowMajor> meshLineEnergy;
    SparseMatrix<double, RowMajor> boundaryEnergy;
    double lastCost;
    double lastLineCost;
    const double INF = 10e8;
    const double PI = 3.1415926535;
    const double EPSILON = 1e-4;
    const cv::Scalar RED = cv::Scalar(0, 0, 255);
    const cv::Scalar BLUE = cv::Scalar(255, 0, 0);
    const cv::Scalar GREEN = cv::Scalar(0, 255, 0);
    const double lambdaB = 10e8;
    const double lambdaL = 100;

};

