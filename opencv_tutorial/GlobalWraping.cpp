#include "GlobalWraping.h"
extern "C"
{
    #include "lsd.h";
}
GlobalWraping::GlobalWraping(Mat& _image, Mat& _mask, Point** mesh, int _meshRows, int _meshCols)
    : image(_image), mask(_mask), meshRows(_meshRows), meshCols(_meshCols)
{
    cols = image.cols;
    rows = image.rows;
    meshVertex = new Coordinate * [meshRows + 1];
    meshLineSegment = new vector<LineSegment> * [meshRows];
    meshLineBilinearWeight = new vector<MatrixXd> * [meshRows];
    meshLineRadianBin = new vector<pair<double, int>>* [meshRows];
    for (int i = 0; i <= meshRows; i++) {
        if (i < meshRows) {
            meshLineSegment[i] = new vector<LineSegment>[meshCols];
            meshLineBilinearWeight[i] = new vector<MatrixXd>[meshCols];
            meshLineRadianBin[i] = new vector<pair<double, int>> [meshCols];
        }
        meshVertex[i] = new Coordinate[meshCols + 1];
        for (int j = 0; j <= meshCols; j++) {
            meshVertex[i][j].col = mesh[i][j].x;
            meshVertex[i][j].row = mesh[i][j].y;
        }
    }
    meshShapeEnergy.resize(8 * meshRows * meshCols, 8 * meshRows * meshCols);
    meshToVertex.resize(8 * meshRows * meshCols, 2 * (meshRows + 1) * (meshCols + 1));
    boundaryEnergy.resize(2 * (meshRows + 1) * (meshCols + 1), 2 * (meshRows + 1) * (meshCols + 1));
    B = VectorXd::Zero(2 * (meshRows + 1) * (meshCols + 1));

}

void GlobalWraping::calcMeshToVertex()
{
    int rowOffset = 0, colOffset = 0, nextColOffset = 2 * (meshCols + 1);
    for (int i = 0; i < meshRows; i++) {
        for (int j = 0; j < meshCols; j++) {
            meshToVertex.insert(rowOffset, colOffset) = 1;
            meshToVertex.insert(rowOffset + 1, colOffset + 1) = 1;
            meshToVertex.insert(rowOffset + 2, colOffset + 2) = 1;
            meshToVertex.insert(rowOffset + 3, colOffset + 3) = 1;
            meshToVertex.insert(rowOffset + 4, nextColOffset) = 1;
            meshToVertex.insert(rowOffset + 5, nextColOffset + 1) = 1;
            meshToVertex.insert(rowOffset + 6, nextColOffset + 2) = 1;
            meshToVertex.insert(rowOffset + 7, nextColOffset + 3) = 1;
            rowOffset += 8;
            colOffset += 2;
            nextColOffset += 2;
        }
        colOffset += 2;
        nextColOffset += 2;
    }
}

void GlobalWraping::calcMeshShapeEnergy()
{
    meshShapeEnergy.setZero();
    int offset = 0;
    MatrixXd A_q(8, 4), temp(8, 8);
    MatrixXd I = MatrixXd::Identity(8, 8);
    for (int i = 0; i < 8; i++) {
        A_q(i, 2) = (i + 1) % 2;
        A_q(i, 3) = i % 2;
    }
    for (int i = 0; i < meshRows; i++) {
        for (int j = 0; j < meshCols; j++) {
            A_q(0, 0) = A_q(1, 1) = meshVertex[i][j].col;
            A_q(0, 1) = -(A_q(1, 0) = meshVertex[i][j].row);
            A_q(2, 0) = A_q(3, 1) = meshVertex[i][j + 1].col;
            A_q(2, 1) = -(A_q(3, 0) = meshVertex[i][j + 1].row);
            A_q(4, 0) = A_q(5, 1) = meshVertex[i + 1][j].col;
            A_q(4, 1) = -(A_q(5, 0) = meshVertex[i + 1][j].row);
            A_q(6, 0) = A_q(7, 1) = meshVertex[i + 1][j + 1].col;
            A_q(6, 1) = -(A_q(7, 0) = meshVertex[i + 1][j + 1].row);
            temp = A_q * (A_q.transpose() * A_q).inverse() * A_q.transpose() - I;
            for (int m = 0; m < 8; m++) {
                for (int n = 0; n < 8; n++) {
                    meshShapeEnergy.insert(offset + m, offset + n) = temp(m, n);
                }
            }
            offset += 8;
        }
    }
    meshShapeEnergy.makeCompressed();
}

void GlobalWraping::calcBoundaryEnergy()
{
    for (int i = 0; i <= meshCols; i++) {
        boundaryEnergy.insert(i * 2 + 1, i * 2 + 1) = 1;
        boundaryEnergy.insert(i * 2 + 2 * meshRows * (meshCols + 1) + 1, i * 2 + 2 * meshRows * (meshCols + 1) + 1) = 1;
        B(i * 2 + 1) = 0;
        B(i * 2 + 2 * meshRows * (meshCols + 1) + 1) = cols - 1;
    }
    for (int i = 0; i <= meshRows; i++) {
        boundaryEnergy.insert(2 * i * (meshCols + 1), 2 * i * (meshCols + 1)) = 1;
        boundaryEnergy.insert(2 * (i + 1) * (meshCols + 1) - 2, 2 * (i + 1) * (meshCols + 1) - 2) = 1;
        B(2 * i * (meshCols + 1)) = 0;
        B(2 * (i + 1) * (meshCols + 1) - 2) = rows - 1;
    }
    boundaryEnergy.makeCompressed();
}

void GlobalWraping::detectLineSegment()
{
    Mat grayImage;
    int lineNumber;
    double* line;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    grayImage.convertTo(grayImage, CV_64F);
    line = lsd(&lineNumber, (double*)grayImage.data, cols, rows);
    vector<LineSegment> lineSegments;

    Mat tempMask;
    mask.copyTo(tempMask);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mask.at<uchar>(i, j) == 0) {
                circle(tempMask, Point(j, i), 2, 0);
            }
        }
    }
    for (int i = 0; i < lineNumber; i++) {
        if (tempMask.at<uchar>(line[i * 7 + 1], line[i * 7 + 0]) != 0 || tempMask.at<uchar>(line[i * 7 + 3], line[i * 7 + 2]) != 0) {
            lineSegments.push_back(LineSegment(line[i * 7], line[i * 7 + 1], line[i * 7 + 2], line[i * 7 + 3]));
        }
    }
#ifdef SHOW_LSD
    Mat tempImage;
    image.copyTo(tempImage);
    for (int i = 0; i < lineSegments.size(); i++) {
        cv::line(tempImage, lineSegments[i].toPoint1(), lineSegments[i].toPoint2(), GREEN);
    }
    imshow("lsd", tempImage);
    waitKey(0);
#endif // SHOW_LSD

    cutLineSegment(lineSegments);
    calcBilinearWeight();
    calcRadianBin();
}

void GlobalWraping::cutLineSegment(vector<LineSegment>& lineSegments)
{
    vector<Coordinate> intersectionPoints;
    Coordinate topLeft, topRight, bottomLeft, bottomRight, linePoint1, linePoint2;
    bool linePoint1InQuad, linePoint2InQuad;
    LineSegment line, newLine;
    for (int i = 0; i < meshRows; i++) {
        for (int j = 0; j < meshCols; j++) {
            topLeft = meshVertex[i][j];
            topRight = meshVertex[i][j + 1];
            bottomLeft = meshVertex[i + 1][j];
            bottomRight = meshVertex[i + 1][j + 1];
            for (int k = 0; k < lineSegments.size(); k++) {
                line = lineSegments[k];
                linePoint1 = Coordinate(line.col1, line.row1);
                linePoint2 = Coordinate(line.col2, line.row2);
                linePoint1InQuad = inQuad(linePoint1, topLeft, topRight, bottomLeft, bottomRight);
                linePoint2InQuad = inQuad(linePoint2, topLeft, topRight, bottomLeft, bottomRight);
                if (linePoint1InQuad && linePoint2InQuad) {
                    meshLineSegment[i][j].push_back(line);
                }
                else if (linePoint1InQuad) {
                    intersectionPoints = getIntersectionWithQuad(line, topLeft, topRight, bottomLeft, bottomRight);
                    if (intersectionPoints.size() != 0) {
                        newLine.col1 = linePoint1.col;
                        newLine.row1 = linePoint1.row;
                        newLine.col2 = intersectionPoints[0].col;
                        newLine.row2 = intersectionPoints[0].row;
                    }
                    assert(inQuad(newLine.toCord1(), topLeft, topRight, bottomLeft, bottomRight));
                    assert(inQuad(newLine.toCord2(), topLeft, topRight, bottomLeft, bottomRight));
                    meshLineSegment[i][j].push_back(newLine);
                }
                else if (linePoint2InQuad) {
                    intersectionPoints = getIntersectionWithQuad(line, topLeft, topRight, bottomLeft, bottomRight);
                    if (intersectionPoints.size() != 0) {
                        newLine.col1 = linePoint2.col;
                        newLine.row1 = linePoint2.row;
                        newLine.col2 = intersectionPoints[0].col;
                        newLine.row2 = intersectionPoints[0].row;
                    }
                    assert(inQuad(newLine.toCord1(), topLeft, topRight, bottomLeft, bottomRight));
                    assert(inQuad(newLine.toCord2(), topLeft, topRight, bottomLeft, bottomRight));
                    meshLineSegment[i][j].push_back(newLine);
                }
                else{
                    intersectionPoints = getIntersectionWithQuad(line, topLeft, topRight, bottomLeft, bottomRight);
                    if (intersectionPoints.size() == 2) {
                        newLine.col1 = intersectionPoints[0].col;
                        newLine.row1 = intersectionPoints[0].row;
                        newLine.col2 = intersectionPoints[1].col;
                        newLine.row2 = intersectionPoints[1].row;
                        assert(inQuad(newLine.toCord1(), topLeft, topRight, bottomLeft, bottomRight));
                        assert(inQuad(newLine.toCord2(), topLeft, topRight, bottomLeft, bottomRight));
                        meshLineSegment[i][j].push_back(newLine);
                    }
                }
            }
        }
    }
}

bool GlobalWraping::inQuad(Coordinate point, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight)
{
    // the point must be to the right of the left line, below the top line, above the bottom line,
    // and to the left of the right line

    // must be right of left line
    if (topLeft.col == bottomLeft.col) {
        if (point.col < topLeft.col - EPSILON) {
            return false;
        }
    }
    else {
        double leftSlope = (topLeft.row - bottomLeft.row) / (topLeft.col - bottomLeft.col);
        double leftIntersect = topLeft.row - leftSlope * topLeft.col;
        double yOnLineX = (point.row - leftIntersect) / leftSlope;
        if (point.col < yOnLineX - EPSILON) {
            return false;
        }
    }
    // must be left of right line
    if (topRight.col == bottomRight.col) {
        if (point.col > topRight.col + EPSILON) {
            return false;
        }
    }
    else {
        double rightSlope = (topRight.row - bottomRight.row) / (topRight.col - bottomRight.col);
        double rightIntersect = topRight.row - rightSlope * topRight.col;
        double yOnLineX = (point.row - rightIntersect) / rightSlope;
        if (point.col > yOnLineX + EPSILON) {
            return false;
        }
    }
    // must be below top line
    if (topLeft.row == topRight.row) {
        if (point.row < topLeft.row - EPSILON) {
            return false;
        }
    }
    else {
        double topSlope = (topRight.row - topLeft.row) / (topRight.col - topLeft.col);
        double topIntersect = topLeft.row - topSlope * topLeft.col;
        double xOnLineY = topSlope * point.col + topIntersect;
        if (point.row < xOnLineY - EPSILON) {
            return false;
        }
    }
    // must be above bottom line
    if (bottomLeft.row == bottomRight.row) {
        if (point.row > bottomLeft.row + EPSILON) {
            return false;
        }
    }
    else {
        double bottomSlope = (bottomRight.row - bottomLeft.row) / (bottomRight.col - bottomLeft.col);
        double bottomIntersect = bottomLeft.row - bottomSlope * bottomLeft.col;
        double xOnLineY = bottomSlope * point.col + bottomIntersect;
        if (point.row > xOnLineY + EPSILON) {
            return false;
        }
    }
    // if all four constraints are satisfied, the point must be in the quad
    return true;
}

vector<Coordinate> GlobalWraping::getIntersectionWithQuad(LineSegment line, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight)
{
    vector<Coordinate> intersections;

    // left
    double leftSlope = INF;
    if (topLeft.col != bottomLeft.col) {
        leftSlope = (topLeft.row - bottomLeft.row) / (topLeft.col - bottomLeft.col);
    }
    double leftIntersect = topLeft.row - leftSlope * topLeft.col;
    // check
    Coordinate leftIntersectPoint;
    if (lineIntersectLine(line, leftSlope, leftIntersect, true, leftIntersectPoint)) {
        if (leftIntersectPoint.row >= topLeft.row && leftIntersectPoint.row <= bottomLeft.row) {
            intersections.push_back(leftIntersectPoint);
        }
    }

    // right
    double rightSlope = INF;
    if (topRight.col != bottomRight.col) {
        rightSlope = (topRight.row - bottomRight.row) / (topRight.col - bottomRight.col);
    }
    double rightIntersect = topRight.row - rightSlope * topRight.col;
    // check
    Coordinate rightIntersectPoint;
    if (lineIntersectLine(line, rightSlope, rightIntersect, true, rightIntersectPoint)) {
        if (rightIntersectPoint.row >= topRight.row && rightIntersectPoint.row <= bottomRight.row) {
            intersections.push_back(rightIntersectPoint);
        }
    }

    // top
    double topSlope = INF;
    if (topLeft.col != topRight.col) {
        topSlope = (topRight.row - topLeft.row) / (topRight.col - topLeft.col);
    }
    double topIntersect = topLeft.row - topSlope * topLeft.col;
    // check
    Coordinate topIntersectPoint;
    if (lineIntersectLine(line, topSlope, topIntersect, false, topIntersectPoint)) {
        if (topIntersectPoint.col >= topLeft.col && topIntersectPoint.col <= topRight.col) {
            intersections.push_back(topIntersectPoint);
        }
    }

    // bottom
    double bottomSlope = INF;
    if (bottomLeft.col != bottomRight.col) {
        bottomSlope = (bottomRight.row - bottomLeft.row) / (bottomRight.col - bottomLeft.col);
    }
    double bottomIntersect = bottomLeft.row - bottomSlope * bottomLeft.col;
    // check
    Coordinate bottomIntersectPoint;
    if (lineIntersectLine(line, bottomSlope, bottomIntersect, false, bottomIntersectPoint)) {
        if (bottomIntersectPoint.col >= bottomLeft.col && bottomIntersectPoint.col <= bottomRight.col) {
            intersections.push_back(bottomIntersectPoint);
        }
    }

    return intersections;
}

bool GlobalWraping::lineIntersectLine(LineSegment line, double slope, double intersect, bool vertical, Coordinate& intersectPoint)
{
    // calculate line segment m and b
    double lineSegmentSlope = INF;
    if (fabs(line.col1 - line.col2) > EPSILON) {
        lineSegmentSlope = (line.row2 - line.row1) / (line.col2 - line.col1);
    }
    double lineSegmentIntersect = line.row1 - lineSegmentSlope * line.col1;

    // calculate intersection
    if (fabs(lineSegmentSlope - slope) < EPSILON) {
        if (fabs(lineSegmentIntersect - intersect) < EPSILON) {
            // same line
            intersectPoint.col = line.col1;
            intersectPoint.row = line.row1;
            return true;
        }
        else {
            return false;
        }
    }
    double intersectX = (intersect - lineSegmentIntersect) / (lineSegmentSlope - slope);
    double intersectY = lineSegmentSlope * intersectX + lineSegmentIntersect;
    if (((intersectX <= line.col1 && intersectX >= line.col2) ||
        (intersectX <= line.col2 && intersectX >= line.col1)) &&
        ((intersectY <= line.row1 && intersectY >= line.row2) ||
        (intersectY <= line.row2 && intersectY >= line.row1))) {
        intersectPoint.col = intersectX;
        intersectPoint.row = intersectY;
        return true;
    }
    else {
        return false;
    }
}

BilinearWeight GlobalWraping::getBilinearWeight(Coordinate point, Coordinate topLeft, Coordinate topRight, Coordinate bottomLeft, Coordinate bottomRight)
{
    /*
    * 
    * X(u,v) = A + (B-A)*u + (C-A)*v + (A-B+D-C)*u*v
    * 
    * X(u,v)-A = (B-A)*u + (C-A)*v + (A-B+D-C)*u*v
    * --------   -----     -----     ---------
    *     ^        ^         ^           ^
    *     H        E         F           G
    * 
    * H = E*u + F*v + G*u*v
    * use x and y to solve
    * 
    * 
    * 
    */
    Coordinate E = topRight - topLeft;
    Coordinate F = bottomLeft - topLeft;
    Coordinate G = topLeft - topRight + bottomRight - bottomLeft;
    Coordinate H = point - topLeft;
    double k2 = G.col * F.row - G.row * F.col;
    double k1 = E.col * F.row - E.row * F.col + H.col * G.row - H.row * G.col;
    double k0 = H.col * E.row - H.row * E.col;
    double delta = k1 * k1 - 4.0 * k2 * k0;
    double u1, v1, u2, v2, u, v;
    assert(delta >= 0.0);
    // parallelogram
    if (fabs(k2) < EPSILON) {
        v = -k0 / k1;
        if (fabs(E.col + G.col * v) < EPSILON) {
            u = (H.row - F.row * v) / (E.row + G.row * v);
        }
        else {
            u = (H.col - F.col * v) / (E.col + G.col * v);
        }
        if (fabs(u - 1) < EPSILON) {
            u = 1;
        }
        else if (fabs(u) < EPSILON) {
            u = 0;
        }
        if (fabs(v - 1) < EPSILON) {
            v = 1;
        }
        else if (fabs(v) < EPSILON) {
            v = 0;
        }
        assert(v >= 0 && v <= 1 && u >= 0 && u <= 1);
    }
    else {
        v1 = (-k1 + sqrt(delta)) / (2.0 * k2);
        v2 = (-k1 - sqrt(delta)) / (2.0 * k2);
        if (v1 >= 0 && v1 <= 1) {
            v = v1;
        }
        else if (v2 >= 0 && v2 <= 1) {
            v = v2;
        }
        else if (fabs(v1) < EPSILON || fabs(v2) < EPSILON) {
            v = 0;
        }
        else if (fabs(v1 - 1) < EPSILON || fabs(v2 - 1) < EPSILON) {
            v = 1;
        }
        else {
            assert(false);
        }
        if (fabs(E.col + G.col * v) < EPSILON) {
            u = (H.row - F.row * v) / (E.row + G.row * v);
        }
        else {
            u = (H.col - F.col * v) / (E.col + G.col * v);
        }
        if (fabs(u - 1) < EPSILON) {
            u = 1;
        }
        else if (fabs(u) < EPSILON) {
            u = 0;
        }
        assert(v >= 0 && v <= 1 && u >= 0 && u <= 1);
    }
    return BilinearWeight(u, v);
}

void GlobalWraping::calcBilinearWeight()
{
    Coordinate topLeft, topRight, bottomLeft, bottomRight, linePoint1, linePoint2;
    LineSegment tempLine;
    double coefA1, coefB1, coefC1, coefD1, coefA2, coefB2, coefC2, coefD2, radian;
    BilinearWeight bilinearWeight1, bilinearWeight2;
    MatrixXd bilinearInterpolation = MatrixXd::Zero(2, 8), bilinearInterpolation2 = MatrixXd::Zero(2, 8);
    for (int i = 0; i < meshRows; i++) {
        for (int j = 0; j < meshCols; j++) {
            topLeft = meshVertex[i][j];
            topRight = meshVertex[i][j + 1];
            bottomLeft = meshVertex[i + 1][j];
            bottomRight = meshVertex[i + 1][j + 1];
            for (int k = 0; k < meshLineSegment[i][j].size(); k++)
            {
                tempLine = meshLineSegment[i][j][k];
                linePoint1 = Coordinate(tempLine.col1, tempLine.row1);
                linePoint2 = Coordinate(tempLine.col2, tempLine.row2);
                bilinearWeight1 = getBilinearWeight(linePoint1, topLeft, topRight, bottomLeft, bottomRight);
                bilinearWeight2 = getBilinearWeight(linePoint2, topLeft, topRight, bottomLeft, bottomRight);
                coefD1 = bilinearWeight1.u * bilinearWeight1.v;
                coefC1 = bilinearWeight1.v - coefD1;
                coefB1 = bilinearWeight1.u - coefD1;
                coefA1 = 1.0 - bilinearWeight1.u - bilinearWeight1.v + coefD1;
                coefD2 = bilinearWeight2.u * bilinearWeight2.v;
                coefC2 = bilinearWeight2.v - coefD2;
                coefB2 = bilinearWeight2.u - coefD2;
                coefA2 = 1.0 - bilinearWeight2.u - bilinearWeight2.v + coefD2;
                bilinearInterpolation(0, 0) = coefA1 - coefA2;
                bilinearInterpolation(0, 2) = coefB1 - coefB2;
                bilinearInterpolation(0, 4) = coefC1 - coefC2;
                bilinearInterpolation(0, 6) = coefD1 - coefD2;
                bilinearInterpolation(1, 1) = coefA1 - coefA2;
                bilinearInterpolation(1, 3) = coefB1 - coefB2;
                bilinearInterpolation(1, 5) = coefC1 - coefC2;
                bilinearInterpolation(1, 7) = coefD1 - coefD2;
                meshLineBilinearWeight[i][j].push_back(bilinearInterpolation);
            }
        }
    }
}

void GlobalWraping::calcRadianBin()
{
    double radian, binWidth = PI / 50;
    int bin;
    VectorXd quadVector = VectorXd::Zero(8), lineVector;
    for (int i = 0; i < meshRows; i++) {
        for (int j = 0; j < meshCols; j++) {
            quadVector(0) = meshVertex[i][j].col;
            quadVector(1) = meshVertex[i][j].row;
            quadVector(2) = meshVertex[i][j + 1].col;
            quadVector(3) = meshVertex[i][j + 1].row;
            quadVector(4) = meshVertex[i + 1][j].col;
            quadVector(5) = meshVertex[i + 1][j].row;
            quadVector(6) = meshVertex[i + 1][j + 1].col;
            quadVector(7) = meshVertex[i + 1][j + 1].row;
            for (int k = 0; k < meshLineSegment[i][j].size(); k++)
            {
                lineVector = meshLineBilinearWeight[i][j][k] * quadVector;
                radian = atan(lineVector(1) / lineVector(0)) + PI / 2;
                bin = floor(radian / binWidth);
                assert(radian >= 0 && radian <= PI);
                assert(bin >= 0 && bin < 50);
                meshLineRadianBin[i][j].push_back(make_pair(radian, bin));
            }
        }
    }
}

void GlobalWraping::drawMesh(bool drawLine)
{
    Mat tempImage;
    image.copyTo(tempImage);
    for (int i = 0; i < meshRows; i++) {
        for (int j = 0; j < meshCols; j++) {
            line(tempImage, meshVertex[i][j].toPoint(), meshVertex[i + 1][j].toPoint(), GREEN);
            line(tempImage, meshVertex[i][j].toPoint(), meshVertex[i][j + 1].toPoint(), GREEN);
        }
        line(tempImage, meshVertex[i][meshCols].toPoint(), meshVertex[i + 1][meshCols].toPoint(), GREEN);
    }
    for (int j = 0; j < meshCols; j++) {
        line(tempImage, meshVertex[meshRows][j].toPoint(), meshVertex[meshRows][j + 1].toPoint(), GREEN);
    }
    //if (drawLine) {
    //    for (int i = 0; i < meshRows; i++) {
    //        for (int j = 0; j < meshCols; j++) {
    //            cout << "mesh: " << i << ", " << j << endl;
    //            for (int k = 0; k < meshLineSegment[i][j].size(); k++) {
    //                line(tempImage, meshLineSegment[i][j][k].toPoint1(), meshLineSegment[i][j][k].toPoint2(), BLUE);
    //            }
    //            imshow("mesh", tempImage);
    //            waitKey(0);
    //        }
    //    }
    //}
    if (drawLine) {
        VectorXd quadVector = VectorXd::Zero(8), lineVector;
        for (int i = 0; i < meshRows; i++) {
            for (int j = 0; j < meshCols; j++) {
                cout << "mesh: " << i << ", " << j << endl;
                quadVector(0) = meshVertex[i][j].col;
                quadVector(1) = meshVertex[i][j].row;
                quadVector(2) = meshVertex[i][j + 1].col;
                quadVector(3) = meshVertex[i][j + 1].row;
                quadVector(4) = meshVertex[i + 1][j].col;
                quadVector(5) = meshVertex[i + 1][j].row;
                quadVector(6) = meshVertex[i + 1][j + 1].col;
                quadVector(7) = meshVertex[i + 1][j + 1].row;
                for (int k = 0; k < meshLineSegment[i][j].size(); k++) {
                    line(tempImage, meshLineSegment[i][j][k].toPoint1(), meshLineSegment[i][j][k].toPoint2(), BLUE);
                }
                imshow("mesh", tempImage);
                waitKey(0);
            }
        }
    }
    imshow("mesh", tempImage);
    waitKey(0);
}

