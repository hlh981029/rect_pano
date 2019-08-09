#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>


using namespace std;
using namespace cv;
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
    const Point toPoint() {
        return Point(col, row);
    }
    friend const Coordinate operator-(const Coordinate& p1, const Coordinate& p2) {
        return Coordinate(p1.col - p2.col, p1.row - p2.row);
    }
    friend const Coordinate operator+(const Coordinate& p1, const Coordinate& p2) {
        return Coordinate(p1.col + p2.col, p1.row + p2.row);
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
    const Point toPoint1() {
        return Point(col1, row1);
    }
    const Point toPoint2() {
        return Point(col2, row2);
    }
    const Coordinate toCord1() {
        return Coordinate(col1, row1);
    }
    const Coordinate toCord2() {
        return Coordinate(col2, row2);
    }
};

class GlobalWraping
{
public:
    GlobalWraping(Mat& _image, Mat& _mask, Point** mesh, int _meshRows, int _meshCols);
    void calcMeshToVertex();
    void calcMeshShapeEnergy();
    void calcBoundaryEnergy();
    void detectLineSegment();
    void cutLineSegment(vector<LineSegment>& lineSegments);
    void calcBilinearWeight();
    void calcRadianBin();
    bool inQuad(Coordinate point, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight);
    vector<Coordinate> getIntersectionWithQuad(LineSegment line, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight);
    bool lineIntersectLine(LineSegment line, double slope, double intersect, bool vertical, Coordinate& intersectPoint);
    BilinearWeight getBilinearWeight(Coordinate point, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight);
    void drawMesh(bool drawLine = false);
    Mat& image;
    Mat& mask;
    int cols, rows, meshRows, meshCols;
    Coordinate** meshVertex;
    int*** meshArray;
    vector<LineSegment>** meshLineSegment;
    vector<MatrixXd>** meshLineBilinearWeight;
    vector<pair<MatrixXd, MatrixXd>>** meshLinePointBilinearWeight;
    vector<pair<double, int>>** meshLineRadianBin;
    VectorXd B;
    SparseMatrix<double> meshToVertex;
    SparseMatrix<double> meshShapeEnergy;
    SparseMatrix<double> boundaryEnergy;
    const double INF = 10e8;
    const double PI = 3.1415926535;
    const double EPSILON = 1e-4;
    const Scalar RED = Scalar(0, 0, 255);
    const Scalar BLUE = Scalar(255, 0, 0);
    const Scalar GREEN = Scalar(0, 255, 0);


};

