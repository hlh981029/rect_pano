#include "SeamCarving.h"
const int MAX_COST = 1e5;

SeamCarving::SeamCarving(cv::Mat& _image, cv::Mat& _mask) :image(_image), mask(_mask)
{
    cv::Mat bgrImage;
    image.convertTo(bgrImage, CV_32FC3, 1.0 / 255);
    rows = image.rows;
    cols = image.cols;
    maxLen = max(cols, rows);
    cvtColor(bgrImage, grayImage, cv::COLOR_BGR2GRAY);
#ifdef USE_RGB
    image.copyTo(expandImage);
    expandImageArray = expandImage.data;
#endif // USE_RGB
#ifdef SHOW_COST
    image.copyTo(seamImage);
#endif // SHOW_COST
#ifdef USE_GRAY
    grayImage.copyTo(expandGrayImage);
    expandGrayArray = (float*)expandGrayImage.data;
#endif // USE_GRAY
    mask.copyTo(expandMaskImage);
    expandMaskArray = expandMaskImage.data;
    displacementIndex.create(rows, cols, CV_32S);
    M.create(maxLen, maxLen, CV_64F);
    route.create(maxLen, maxLen, CV_32S);
    M.setTo(0);
    neighborIndexArray = new int* [maxLen];
    for (int i = 0; i < maxLen; i++) {
        neighborIndexArray[i] = new int[5];
    }
    maskArray = mask.data;
    mArray = (double*)M.data;
    routeArray = (int*)route.data;
    displacementIndexArray = (int*)displacementIndex.data;
    expandMaskRowArray = expandMaskArray;
    displacementIndexRowArray = displacementIndexArray;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            displacementIndexRowArray[j] = i * cols + j;
            if (expandMaskRowArray[j] != 255) {
                expandMaskRowArray[j] = 0;
            }
            else {
                expandMaskRowArray[j] = 16;
            }
        }
        expandMaskRowArray += cols;
        displacementIndexRowArray += cols;
    }
    directionMask = Vertical;
}

BoundarySegment SeamCarving::getLongestBoundary()
{
    int start, end, length, maxLength = 0, maxStart = 0, maxEnd = 0;
    Direction maxDirection = Direction::None;
    uchar* boundaryMask;
    // Left
    start = end = length = 0;
    boundaryMask = expandMaskArray;
    while (end < rows) {
        if (boundaryMask[start * cols] != 0) {
            end = start = start + 1;
        }
        else {
            if (end + 1 < rows && boundaryMask[(end + 1) * cols] == 0) {
                end++;
            }
            else {
                length = end - start + 1;
                if (length > maxLength) {
                    maxLength = length;
                    maxStart = start;
                    maxEnd = end;
                    maxDirection = Direction::Left;
                }
                start = end = end + 1;
            }
        }
    }
    // Right
    start = end = length = 0;
    boundaryMask = expandMaskArray + cols - 1;
    while (end < rows) {
        if (boundaryMask[start * cols] != 0) {
            end = start = start + 1;
        }
        else {
            if (end + 1 < rows && boundaryMask[(end + 1) * cols] == 0) {
                end++;
            }
            else {
                length = end - start + 1;
                if (length > maxLength) {
                    maxLength = length;
                    maxStart = start;
                    maxEnd = end;
                    maxDirection = Direction::Right;
                }
                start = end = end + 1;
            }
        }
    }
    // Top
    start = end = length = 0;
    boundaryMask = expandMaskArray;
    while (end < cols) {
        if (boundaryMask[start] != 0) {
            end = start = start + 1;
        }
        else {
            if (end + 1 < cols && boundaryMask[end + 1] == 0) {
                end++;
            }
            else {
                length = end - start + 1;
                if (length > maxLength) {
                    maxLength = length;
                    maxStart = start;
                    maxEnd = end;
                    maxDirection = Direction::Top;
                }
                start = end = end + 1;
            }
        }
    }
    // Bottom
    start = end = length = 0;
    boundaryMask = expandMaskArray + (rows - 1) * cols;
    while (end < cols) {
        if (boundaryMask[start] != 0) {
            end = start = start + 1;
        }
        else {
            if (end + 1 < cols && boundaryMask[end + 1] == 0) {
                end++;
            }
            else {
                length = end - start + 1;
                if (length > maxLength) {
                    maxLength = length;
                    maxStart = start;
                    maxEnd = end;
                    maxDirection = Direction::Bottom;
                }
                start = end = end + 1;
            }
        }
    }

    if (maxLength == 0) {
        return BoundarySegment(0, 0, Direction::None);
    }
    return BoundarySegment(maxStart, maxEnd, maxDirection);
}

void SeamCarving::localWarping()
{
}

void SeamCarving::insertSeam(BoundarySegment boundarySegment)
{
    int begin = boundarySegment.begin, end = boundarySegment.end;
    int min = 0;
    double temp = 1e10;
    expandMaskRowArray = expandMaskArray + end * cols;
    mRowArray = mArray + end * cols;
    for (int i = 0; i < cols; i++) {
        assert(mRowArray[i] < 0 || (mRowArray[i] >= 0 && (expandMaskRowArray[i] & directionMask) == 0));
        if (mRowArray[i] >= 0 && mRowArray[i] < temp) {
            min = i;
            temp = mArray[end * cols + i];
        }
    }
    if (boundarySegment.direction == Bottom || boundarySegment.direction == Right) {
        displacementIndexRowArray = displacementIndexArray + end * cols;
        for (int i = end; i >= begin; i--) {
            assert((expandMaskRowArray[min] & directionMask) == 0);
            assert(min >= 0 && min < cols);
            for (int j = cols - 1; j > min; j--) {
                expandMaskRowArray[j] = expandMaskRowArray[j - 1];
                displacementIndexRowArray[j] = displacementIndexRowArray[j - 1];
#ifdef USE_GRAY
                expandGrayArray[i * cols + j] = expandGrayArray[i * cols + j - 1];
#endif // USE_GRAY
#ifdef USE_RGB
                expandImage.at<cv::Vec3b>(i, j) = expandImage.at<cv::Vec3b>(i, j - 1);
#endif // USE_RGB
#ifdef SHOW_COST
                seamImage.at<cv::Vec3b>(i, j) = seamImage.at<cv::Vec3b>(i, j - 1);
#endif // SHOW_COST
            }
#ifdef SHOW_COST
            seamImage.at<cv::Vec3b>(i, min) = cv::Vec3b(255, 0, 0);
#endif // SHOW_COST
            if (expandMaskRowArray[min] != 0) {
                assert(expandMaskRowArray[min + 1] == expandMaskRowArray[min]);
                expandMaskRowArray[min + 1] = expandMaskRowArray[min] = (expandMaskRowArray[min] | directionMask);
            }
            min = routeArray[i * cols + min];
            displacementIndexRowArray -= cols;
            expandMaskRowArray -= cols;
        }
    }
    else {
        displacementIndexRowArray = displacementIndexArray + end * cols;
        for (int i = end; i >= begin; i--) {
            assert((expandMaskRowArray[min] & directionMask) == 0);
            assert(min >= 0 && min < cols);
            for (int j = 0; j < min; j++) {
                expandMaskRowArray[j] = expandMaskRowArray[j + 1];
                displacementIndexRowArray[j] = displacementIndexRowArray[j + 1];
#ifdef USE_GRAY
                expandGrayArray[i * cols + j] = expandGrayArray[i * cols + j + 1];
#endif // USE_GRAY
#ifdef USE_RGB
                expandImage.at<cv::Vec3b>(i, j) = expandImage.at<cv::Vec3b>(i, j + 1);
#endif // USE_RGB
#ifdef SHOW_COST
                seamImage.at<cv::Vec3b>(i, j) = seamImage.at<cv::Vec3b>(i, j + 1);
#endif // SHOW_COST
            }
#ifdef SHOW_COST
            seamImage.at<cv::Vec3b>(i, min) = cv::Vec3b(255, 0, 0);
#endif // SHOW_COST
            if (expandMaskRowArray[min] != 0) {
                assert(expandMaskRowArray[min - 1] == expandMaskRowArray[min]);
                expandMaskRowArray[min - 1] = expandMaskRowArray[min] = (expandMaskRowArray[min] | directionMask);
            }
            min = routeArray[i * cols + min];
            displacementIndexRowArray -= cols;
            expandMaskRowArray -= cols;
        }
    }
    if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
#ifdef USE_RGB
        expandImage = expandImage.t();
        expandImageArray = expandImage.data;
#endif // USE_RGB
#ifdef USE_GRAY
        expandGrayImage = expandGrayImage.t();
        expandGrayArray = (float*)expandGrayImage.data;
#endif // USE_GRAY
#ifdef SHOW_COST
        seamImage = seamImage.t();
#endif // SHOW_COST
        expandMaskImage = expandMaskImage.t();
        expandMaskArray = expandMaskImage.data;
        displacementIndex = displacementIndex.t();
        displacementIndexArray = (int*)displacementIndex.data;

        swap(cols, rows);
        directionMask = Vertical;
    }
}

void SeamCarving::calcCost(BoundarySegment boundarySegment)
{
    if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
#ifdef USE_RGB
        expandImage = expandImage.t();
        expandImageArray = expandImage.data;
#endif // USE_RGB
#ifdef USE_GRAY
        expandGrayImage = expandGrayImage.t();
        expandGrayArray = (float*)expandGrayImage.data;
#endif // USE_GRAY
#ifdef SHOW_COST
        seamImage = seamImage.t();
#endif // SHOW_COST
        expandMaskImage = expandMaskImage.t();
        expandMaskArray = expandMaskImage.data;
        displacementIndex = displacementIndex.t();
        displacementIndexArray = (int*)displacementIndex.data;
        swap(cols, rows);
        directionMask = Horizontal;
    }
    begin = boundarySegment.begin;
    end = boundarySegment.end;
    direction = boundarySegment.direction;
    M.setTo(-1);
    route.setTo(-1);
    rowOffset = begin * cols;
    mRowArray = mArray + rowOffset;
    routeRowArray = routeArray + rowOffset;
    expandMaskRowArray = expandMaskArray + rowOffset;
    mUpRowArray = mRowArray - cols;
    routeUpRowArray = routeRowArray - cols;
    expandMaskUpRowArray = expandMaskRowArray - cols;
#ifdef USE_RGB
    expandImageRowArray = expandImageArray + rowOffset * 3;
    expandImageUpRowArray = expandImageRowArray - cols * 3;
#endif // USE_RGB
    for (int i = begin; i <= end; i++) {
        if (i == begin) {
            if ((expandMaskRowArray[0] & directionMask) == 0) {
                left = 0;
            }
            else {
                left = -1;
            }
            if ((expandMaskRowArray[cols - 1] & directionMask) == 0) {
                right = cols - 1;
            }
            else {
                right = cols;
            }
            for (int j = 0; j < cols; j++) {
                neighborIndexArray[j][0] = left;
                neighborIndexArray[cols - 1 - j][1] = right;
                if ((expandMaskRowArray[j] & directionMask) == 0) {
                    left = j;
                }
                if ((expandMaskRowArray[cols - 1 - j] & directionMask) == 0) {
                    right = cols - 1 - j;
                }
            }
            for (int j = 0; j < cols; j++) {
                if ((expandMaskRowArray[j] & directionMask) == 0) {
                    left = neighborIndexArray[j][0] == -1 ? j : neighborIndexArray[j][0];
                    right = neighborIndexArray[j][1] == cols ? j : neighborIndexArray[j][1];
                    assert(left < right && left >= 0 && right < cols);
#ifdef USE_GRAY
                    mRowArray[j] = abs(expandGrayArray[i * cols + left] - expandGrayArray[i * cols + right]);
#endif // USE_GRAY
#ifdef USE_RGB
                    mRowArray[j] = abs(expandImageRowArray[left * 3] - expandImageRowArray[right * 3])
                        + abs(expandImageRowArray[left * 3 + 1] - expandImageRowArray[right * 3 + 1])
                        + abs(expandImageRowArray[left * 3 + 2] - expandImageRowArray[right * 3 + 2]);
#endif // USE_RGB
                    routeRowArray[j] = 0;
                    if (expandMaskRowArray[j] == 0) {
                        mRowArray[j] += MAX_COST;
                    }
                }
            }
        }
        else {
            if ((expandMaskRowArray[0] & directionMask) == 0) {
                left = 0;
            }
            else {
                left = -1;
            }
            if ((expandMaskRowArray[cols - 1] & directionMask) == 0) {
                right = cols - 1;
            }
            else {
                right = cols;
            }
            if ((expandMaskUpRowArray[0] & directionMask) == 0) {
                upLeft = 0;
            }
            else {
                upLeft = -1;
            }
            if ((expandMaskUpRowArray[cols - 1] & directionMask) == 0) {
                upRight = cols - 1;
            }
            else {
                upRight = cols;
            }
            for (int j = 0; j < cols; j++) {
                neighborIndexArray[j][0] = left;
                neighborIndexArray[cols - 1 - j][1] = right;
                neighborIndexArray[j][2] = upLeft;
                neighborIndexArray[cols - 1 - j][3] = upRight;
                if ((expandMaskRowArray[j] & directionMask) == 0) {
                    left = j;
                }
                if ((expandMaskRowArray[cols - 1 - j] & directionMask) == 0) {
                    right = cols - 1 - j;
                }
                if ((expandMaskUpRowArray[j] & directionMask) == 0) {
                    upLeft = j;
                }
                if ((expandMaskUpRowArray[cols - 1 - j] & directionMask) == 0) {
                    upRight = cols - 1 - j;
                }
            }
            for (int j = 0; j < cols; j++) {
                if ((expandMaskUpRowArray[j] & directionMask) == 0) {
                    neighborIndexArray[j][4] = j;
                }
                else {
                    upLeft = neighborIndexArray[j][2];
                    upRight = neighborIndexArray[j][3];
                    if (upLeft == -1) {
                        assert(upRight != -1);
                        neighborIndexArray[j][4] = upRight;
                        neighborIndexArray[j][3] = neighborIndexArray[upRight][3];
                    }
                    else if (upRight == -1) {
                        assert(upLeft != -1);
                        neighborIndexArray[j][4] = upLeft;
                        neighborIndexArray[j][2] = neighborIndexArray[upLeft][2];
                    }
                    else {
                        left = upLeft;
                        right = upRight;
                        while (j - left == right - j) {
                            left = neighborIndexArray[left][2];
                            right = neighborIndexArray[right][3];
                        }
                        if (left < right) {
                            neighborIndexArray[j][4] = upLeft;
                            neighborIndexArray[j][2] = neighborIndexArray[upLeft][2];
                        }
                        else {
                            assert(left > right);
                            neighborIndexArray[j][4] = upRight;
                            neighborIndexArray[j][3] = neighborIndexArray[upRight][3];

                        }
                    }
                }
                assert(neighborIndexArray[j][4] >= 0 && neighborIndexArray[j][4] < cols);
            }
            for (int j = 0; j < cols; j++) {
                if ((expandMaskRowArray[j] & directionMask) == 0) {
                    left = neighborIndexArray[j][0] == -1 ? j : neighborIndexArray[j][0];
                    right = neighborIndexArray[j][1] == cols ? j : neighborIndexArray[j][1];
                    upLeft = neighborIndexArray[j][2];
                    upRight = neighborIndexArray[j][3];
                    up = neighborIndexArray[j][4];
                    assert(left < right && left >= 0 && right < cols);
#ifdef USE_GRAY
                    tempUpCost = abs(expandGrayArray[i * cols + left] - expandGrayArray[i * cols + right]);
#endif // USE_GRAY
#ifdef USE_RGB
                    tempUpCost = abs(expandImageRowArray[left * 3] - expandImageRowArray[right * 3])
                        + abs(expandImageRowArray[left * 3 + 1] - expandImageRowArray[right * 3 + 1])
                        + abs(expandImageRowArray[left * 3 + 2] - expandImageRowArray[right * 3 + 2]);
#endif // USE_RGB

                    minCost = tempUpCost + mUpRowArray[up];
                    minCostIndex = up;
                    if (upLeft != -1 && up != upLeft) {
                        tempLeftCost = tempUpCost + mUpRowArray[upLeft]
#ifdef USE_GRAY
                            + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
#ifdef USE_RGB
                            + abs(expandImageUpRowArray[up * 3] - expandImageRowArray[left * 3])
                            + abs(expandImageUpRowArray[up * 3 + 1] - expandImageRowArray[left * 3 + 1])
                            + abs(expandImageUpRowArray[up * 3 + 2] - expandImageRowArray[left * 3 + 2]);
#endif // USE_RGB

                        if (tempLeftCost < minCost) {
                            minCost = tempLeftCost;
                            minCostIndex = upLeft;
                        }
                    }
                    if (upRight != -1 && up != upRight) {
                        tempRightCost = tempUpCost + mUpRowArray[upRight]
#ifdef USE_GRAY
                            + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
#endif // USE_GRAY
#ifdef USE_RGB
                            + abs(expandImageUpRowArray[up * 3] - expandImageRowArray[right * 3])
                            + abs(expandImageUpRowArray[up * 3 + 1] - expandImageRowArray[right * 3 + 1])
                            + abs(expandImageUpRowArray[up * 3 + 2] - expandImageRowArray[right * 3 + 2]);
#endif // USE_RGB
                        if (tempRightCost < minCost) {
                            minCost = tempRightCost;
                            minCostIndex = upRight;
                        }
                    }
                    assert(minCost >= 0);
                    assert(minCostIndex != -1);
                    assert(routeUpRowArray[minCostIndex] != -1);
                    assert((expandMaskUpRowArray[minCostIndex] & directionMask) == 0);
                    if (expandMaskRowArray[j] == 0) {
                        minCost += MAX_COST;
                    }
                    routeRowArray[j] = minCostIndex;
                    mRowArray[j] = minCost;
                }
            }
        }
        mUpRowArray = mRowArray;
        routeUpRowArray = routeRowArray;
        expandMaskUpRowArray = expandMaskRowArray;
        mRowArray += cols;
        routeRowArray += cols;
        expandMaskRowArray += cols;
#ifdef USE_RGB
        expandImageUpRowArray = expandImageRowArray;
        expandImageRowArray += cols * 3;
#endif // USE_RGB
    }
}

void SeamCarving::showCost(BoundarySegment boundarySegment)
{
#ifdef SHOW_COST
    cv::Mat tempImage;
    seamImage.copyTo(tempImage);
    int begin = boundarySegment.begin, end = boundarySegment.end;
    int min = 0;
    double temp = 1e9;
    expandMaskRowArray = expandMaskArray + end * cols;
    mRowArray = mArray + end * cols;
    for (int i = 0; i < cols; i++) {
        assert(mRowArray[i] < 0 || (mRowArray[i] >= 0 && (expandMaskRowArray[i] & directionMask) == 0));
        if (mRowArray[i] >= 0 && mRowArray[i] < temp) {
            min = i;
            temp = mArray[end * cols + i];
        }
    }
    if (boundarySegment.direction == Bottom || boundarySegment.direction == Right) {
        for (int i = end; i >= begin; i--) {
            assert((expandMaskArray[i * cols + min] & directionMask) == 0);
            assert(min >= 0 && min < cols);
            tempImage.at<cv::Vec3b>(i, min) = cv::Vec3b(0, 255, 0);
            min = routeArray[i * cols + min];
            tempImage.at<cv::Vec3b>(i, cols - 1) = cv::Vec3b(0, 0, 255);
        }
    }
    else {
        for (int i = end; i >= begin; i--) {
            assert((expandMaskArray[i * cols + min] & directionMask) == 0);
            assert(min >= 0 && min < cols);
            tempImage.at<cv::Vec3b>(i, min) = cv::Vec3b(0, 255, 0);
            min = routeArray[i * cols + min];
            tempImage.at<cv::Vec3b>(i, 0) = cv::Vec3b(0, 0, 255);
        }
    }
    if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
        tempImage = tempImage.t();
    }
    imshow("seam", tempImage);
    cv::waitKey(0);
#endif // SHOW_COST
}

void SeamCarving::placeMesh()
{
    double ratio, meshRowStep, meshColStep;
    int row, col, index;
    ratio = 1.0 * cols / rows;
    meshRows = round(sqrt(400 / ratio));
    meshCols = round(meshRows * ratio);
    meshRowStep = rows / meshRows;
    meshColStep = cols / meshCols;
    mesh = new cv::Point * [meshRows + 1];
    for (int i = 0; i <= meshRows; i++) {
        mesh[i] = new cv::Point[meshCols + 1];
    }
    for (int i = 0; i <= meshRows; i++) {
        for (int j = 0; j <= meshCols; j++) {
            if (j == meshCols) {
                col = cols - 1;
            }
            else {
                col = round(j * meshColStep);
            }
            if (i == meshRows) {
                row = rows - 1;
            }
            else {
                row = round(i * meshRowStep);
            }
            assert(col < cols && col >= 0);
            assert(row < rows && row >= 0);
            index = displacementIndexArray[row * cols + col];
            assert(index > 0);
            //assert(maskArray[index] == 255);
            mesh[i][j].y = index / cols;
            mesh[i][j].x = index % cols;
        }
    }
}

BoundarySegment::BoundarySegment(int _begin, int _end, Direction _direction) :begin(_begin), end(_end), direction(_direction) { }

void BoundarySegment::print()
{
    cout << direction << ' ' << begin << ' ' << end << endl;
}
