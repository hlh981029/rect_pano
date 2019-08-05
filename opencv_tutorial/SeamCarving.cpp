#include "SeamCarving.h"
const int MAX_COST = 1e5;

SeamCarving::SeamCarving(Mat& _image, Mat& _mask) :image(_image), mask(_mask)
{
    Mat bgrImage;
    image.convertTo(bgrImage, CV_32FC3, 1.0 / 255);
    rows = image.rows;
    cols = image.cols;
    maxLen = max(cols, rows);
    cvtColor(bgrImage, grayImage, COLOR_BGR2GRAY);

    image.copyTo(expandImage);
    mask.copyTo(expandMaskImage);
    grayImage.copyTo(expandGrayImage);
    expandGrayArray = (float*)expandGrayImage.data;
    expandMaskArray = expandMaskImage.data;


    //displacementIndex.create(rows, cols, CV_32S);
    //imageIndexUsed.create(rows, cols, CV_8U);
    M.create(maxLen, maxLen, CV_64F);
    route.create(maxLen, maxLen, CV_32S);
    M.setTo(0);
    neighborIndexArray = new int* [maxLen];
    for (int i = 0; i < maxLen; i++) {
        neighborIndexArray[i] = new int[5];
    }
    //imageIndexUsed.setTo(0);
    maskArray = mask.data;
    mArray = (double*)M.data;
    routeArray = (int*)route.data;
    expandImageArray = expandImage.data;
    //imageIndexUsedArray = imageIndexUsed.data;
    //displacementIndexArray = (int*)displacementIndex.data;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //displacementIndexArray[i * cols + j] = i * cols + j;
            if (expandMaskArray[i * cols + j] != 255) {
                expandMaskArray[i * cols + j] = 0;
            }
            else {
                expandMaskArray[i * cols + j] = 16;
            }
        }
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

void SeamCarving::localWraping()
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
        for (int i = end; i >= begin; i--) {
            assert((expandMaskArray[i * cols + min] & directionMask) == 0);
            assert(min >= 0 && min < cols);
            for (int j = cols - 1; j > min; j--) {
                expandGrayArray[i * cols + j] = expandGrayArray[i * cols + j - 1];
                expandMaskArray[i * cols + j] = expandMaskArray[i * cols + j - 1];
                expandImage.at<Vec3b>(i, j) = expandImage.at<Vec3b>(i, j - 1);
                image.at<Vec3b>(i, j) = image.at<Vec3b>(i, j - 1);
            }
            image.at<Vec3b>(i, min) = Vec3b(255, 0, 0);
            if (expandMaskArray[i * cols + min] != 0) {
                assert(expandMaskArray[i * cols + min + 1] == expandMaskArray[i * cols + min]);
                expandMaskArray[i * cols + min + 1] = (expandMaskArray[i * cols + min + 1] | boundarySegment.direction);
                expandMaskArray[i * cols + min] = (expandMaskArray[i * cols + min] | boundarySegment.direction);
            }
            min = routeArray[i * cols + min];
        }
    }
    else {
        for (int i = end; i >= begin; i--) {
            assert((expandMaskArray[i * cols + min] & directionMask) == 0);
            assert(min >= 0 && min < cols);
            for (int j = 0; j < min; j++) {
                expandGrayArray[i * cols + j] = expandGrayArray[i * cols + j + 1];
                expandMaskArray[i * cols + j] = expandMaskArray[i * cols + j + 1];
                expandImage.at<Vec3b>(i, j) = expandImage.at<Vec3b>(i, j + 1);
                image.at<Vec3b>(i, j) = image.at<Vec3b>(i, j + 1);
            }
            image.at<Vec3b>(i, min) = Vec3b(255, 0, 0);
            if (expandMaskArray[i * cols + min] != 0) {
                assert(expandMaskArray[i * cols + min - 1] == expandMaskArray[i * cols + min]);
                expandMaskArray[i * cols + min - 1] = (expandMaskArray[i * cols + min - 1] | boundarySegment.direction);
                expandMaskArray[i * cols + min] = (expandMaskArray[i * cols + min] | boundarySegment.direction);
            }
            min = routeArray[i * cols + min];

        }
    }
    if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
        expandImage = expandImage.t();
        expandMaskImage = expandMaskImage.t();
        expandGrayImage = expandGrayImage.t();
        expandImageArray = expandImage.data;
        expandMaskArray = expandMaskImage.data;
        expandGrayArray = (float*)expandGrayImage.data;
        image = image.t();
        swap(cols, rows);
        directionMask = Vertical;
    }
}

void SeamCarving::calcCost(BoundarySegment boundarySegment)
{
    if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
        expandImage = expandImage.t();
        expandMaskImage = expandMaskImage.t();
        expandGrayImage = expandGrayImage.t();
        expandImageArray = expandImage.data;
        expandMaskArray = expandMaskImage.data;
        expandGrayArray = (float*)expandGrayImage.data;
        image = image.t();
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
    expandImageRowArray = expandImageArray + rowOffset * 3;
    mUpRowArray = mRowArray - cols;
    routeUpRowArray = routeRowArray - cols;
    expandMaskUpRowArray = expandMaskRowArray - cols;
    expandImageUpRowArray = expandImageRowArray - cols * 3;
    for (int i = begin; i <= end; i++) {
        if (i == begin) {
            for (int j = 0; j < cols; j++) {
                neighborIndexArray[j][0] = j;
            }
            for (int j = 0; j < cols; j++) {
                if ((expandMaskRowArray[neighborIndexArray[j][0]] & (directionMask & (Left | Top))) != 0) {
                    for (int k = j; k > 0; k--) {
                        neighborIndexArray[k][0] = neighborIndexArray[k - 1][0];
                        if (neighborIndexArray[k][0] == -1) {
                            break;
                        }
                        neighborIndexArray[0][0] = -1;
                    }
                }
            }
            for (int j = cols - 1; j >= 0; j--) {
                if (neighborIndexArray[j][0] == -1) {
                    break;
                }
                if ((expandMaskRowArray[neighborIndexArray[j][0]] & (directionMask & (Right | Bottom))) != 0) {
                    for (int k = j; k < cols - 1; k++) {
                        neighborIndexArray[k][0] = neighborIndexArray[k + 1][0];
                        if (neighborIndexArray[k][0] == -1) {
                            break;
                        }
                        neighborIndexArray[cols - 1][0] = -1;
                    }
                }
            }
            for (int j = 0; j < cols; j++) {
                assert(neighborIndexArray[j][0] == -1 || (expandMaskRowArray[neighborIndexArray[j][0]] & directionMask) == 0);
            }
            for (int j = 0; j < cols; j++) {
                middle = neighborIndexArray[j][0];
                if (middle == -1) {
                    continue;
                }
                if (j == 0 || neighborIndexArray[j - 1][0] == -1) {
                    left = middle;
                    right = neighborIndexArray[j + 1][0];
                }
                else if (j == cols - 1 || neighborIndexArray[j + 1][0] == -1) {
                    left = neighborIndexArray[j - 1][0];
                    right = middle;
                }
                else {
                    left = neighborIndexArray[j - 1][0];
                    right = neighborIndexArray[j + 1][0];
                }
                assert((expandMaskRowArray[left] & directionMask) == 0 && (expandMaskRowArray[right] & directionMask) == 0);
                tempUpCost = abs(expandImageRowArray[left * 3] - expandImageRowArray[right * 3])
                    + abs(expandImageRowArray[left * 3 + 1] - expandImageRowArray[right * 3 + 1])
                    + abs(expandImageRowArray[left * 3 + 2] - expandImageRowArray[right * 3 + 2]);
                mRowArray[middle] = tempUpCost;
                routeRowArray[middle] = 0;
            }
        }
        else {
            for (int j = 0; j < cols; j++) {
                neighborIndexArray[j][0] = neighborIndexArray[j][1] = j;
            }
            for (int j = 0; j < cols; j++) {
                if ((expandMaskRowArray[neighborIndexArray[j][0]] & (directionMask & (Left | Top))) != 0) {
                    for (int k = j; k > 0; k--) {
                        neighborIndexArray[k][0] = neighborIndexArray[k - 1][0];
                        if (neighborIndexArray[k][0] == -1) {
                            break;
                        }
                        neighborIndexArray[0][0] = -1;
                    }
                }
                if ((expandMaskUpRowArray[neighborIndexArray[j][1]] & (directionMask & (Left | Top))) != 0) {
                    for (int k = j; k > 0; k--) {
                        neighborIndexArray[k][1] = neighborIndexArray[k - 1][1];
                        if (neighborIndexArray[k][1] == -1) {
                            break;
                        }
                        neighborIndexArray[0][1] = -1;
                    }
                }
            }
            for (int j = cols - 1; j >= 0; j--) {
                if (neighborIndexArray[j][0] == -1 && neighborIndexArray[j][1] == -1) {
                    break;
                }
                if (neighborIndexArray[j][0] != -1 && (expandMaskRowArray[neighborIndexArray[j][0]] & (directionMask & (Right | Bottom))) != 0) {
                    for (int k = j; k < cols - 1; k++) {
                        neighborIndexArray[k][0] = neighborIndexArray[k + 1][0];
                        if (neighborIndexArray[k][0] == -1) {
                            break;
                        }
                        neighborIndexArray[cols - 1][0] = -1;
                    }
                }
                if (neighborIndexArray[j][1] != -1 && (expandMaskUpRowArray[neighborIndexArray[j][1]] & (directionMask & (Right | Bottom))) != 0) {
                    for (int k = j; k < cols - 1; k++) {
                        neighborIndexArray[k][1] = neighborIndexArray[k + 1][1];
                        if (neighborIndexArray[k][1] == -1) {
                            break;
                        }
                        neighborIndexArray[cols - 1][1] = -1;
                    }
                }
            }
            for (int j = 0; j < cols; j++) {
                assert(neighborIndexArray[j][0] == -1 || (expandMaskRowArray[neighborIndexArray[j][0]] & directionMask) == 0);
                assert(neighborIndexArray[j][1] == -1 || (expandMaskUpRowArray[neighborIndexArray[j][1]] & directionMask) == 0);
            }
            for (int j = 0; j < cols; j++) {
                hasLeft = hasRight = hasUp = true;
                left = right = middle = upLeft = upRight = up = -1;
                if (neighborIndexArray[j][0] != -1) {
                    middle = neighborIndexArray[j][0];
                    if (neighborIndexArray[j][1] != -1) {
                        up = neighborIndexArray[j][1];
                        if (j == 0 || neighborIndexArray[j - 1][0] == -1) {
                            hasLeft = false;
                            left = neighborIndexArray[j][0];
                            right = neighborIndexArray[j + 1][0];
                            upRight = neighborIndexArray[j + 1][1];
                            assert(upRight != -1);
                            if (j != 0 && neighborIndexArray[j - 1][1] != -1) {
                                hasLeft = true;
                                upLeft = neighborIndexArray[j - 1][1];
                            }
                        }
                        else if (j == cols - 1 || neighborIndexArray[j + 1][0] == -1) {
                            hasRight = false;
                            left = neighborIndexArray[j - 1][0];
                            right = neighborIndexArray[j][0];
                            upLeft = neighborIndexArray[j - 1][1];
                            assert(upLeft != -1);
                            if (j != cols - 1 && neighborIndexArray[j + 1][1] != -1) {
                                hasRight = true;
                                upRight = neighborIndexArray[j + 1][1];
                            }
                        }
                        else {
                            left = neighborIndexArray[j - 1][0];
                            right = neighborIndexArray[j + 1][0];
                            if (neighborIndexArray[j - 1][1] == -1) {
                                hasLeft = false;
                            }
                            else {
                                upLeft = neighborIndexArray[j - 1][1];
                            }
                            if (neighborIndexArray[j + 1][1] == -1) {
                                hasRight = false;
                            }
                            else {
                                upRight = neighborIndexArray[j + 1][1];
                            }
                        }
                        tempUpCost = abs(expandImageRowArray[left * 3] - expandImageRowArray[right * 3])
                                   + abs(expandImageRowArray[left * 3 + 1] - expandImageRowArray[right * 3 + 1])
                                   + abs(expandImageRowArray[left * 3 + 2] - expandImageRowArray[right * 3 + 2]);
                        minCost = tempUpCost + mUpRowArray[up];
                        minCostIndex = up;
                        if (hasLeft) {
                            tempLeftCost = tempUpCost + mUpRowArray[upLeft]
                                         + abs(expandImageUpRowArray[up * 3] - expandImageRowArray[right * 3])
                                         + abs(expandImageUpRowArray[up * 3 + 1] - expandImageRowArray[right * 3 + 1])
                                         + abs(expandImageUpRowArray[up * 3 + 2] - expandImageRowArray[right * 3 + 2]);
                            if (tempLeftCost < minCost) {
                                minCost = tempLeftCost;
                                minCostIndex = upLeft;
                            }
                        }
                        if (hasRight) {
                            tempRightCost = tempUpCost + mUpRowArray[upRight]
                                          + abs(expandImageUpRowArray[up * 3] - expandImageRowArray[left * 3])
                                          + abs(expandImageUpRowArray[up * 3 + 1] - expandImageRowArray[left * 3 + 1])
                                          + abs(expandImageUpRowArray[up * 3 + 2] - expandImageRowArray[left * 3 + 2]);
                            if (tempRightCost < minCost) {
                                minCost = tempRightCost;
                                minCostIndex = upRight;
                            }
                        }
                        if (minCost > 10e9 - 1) {
                            mRowArray[middle] = 10e9;
                        }
                        else {
                            assert(minCost >= 0);
                            assert(minCostIndex != -1);
                            assert(routeUpRowArray[minCostIndex] != -1);
                            assert((expandMaskUpRowArray[minCostIndex] & directionMask) == 0);
                            if (expandMaskRowArray[middle] == 0) {
                                minCost += MAX_COST;
                            }
                            routeRowArray[middle] = minCostIndex;
                            mRowArray[middle] = minCost;
                        }
                    }
                    else {
                        mRowArray[middle] = 10e9;
                    }
                }
            }
        }
        mUpRowArray = mRowArray;
        routeUpRowArray = routeRowArray;
        expandMaskUpRowArray = expandMaskRowArray;
        expandImageUpRowArray = expandImageRowArray;
        mRowArray += cols;
        routeRowArray += cols;
        expandMaskRowArray += cols;
        expandImageRowArray += cols * 3;
    }
}

void SeamCarving::showCost(BoundarySegment boundarySegment)
{
    Mat tempImage;
    image.copyTo(tempImage);
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
            tempImage.at<Vec3b>(i, min) = Vec3b(0, 255, 0);
            min = routeArray[i * cols + min];
            tempImage.at<Vec3b>(i, cols - 1) = Vec3b(0, 0, 255);
        }
    }
    else {
        for (int i = end; i >= begin; i--) {
            assert((expandMaskArray[i * cols + min] & directionMask) == 0);
            assert(min >= 0 && min < cols);
            tempImage.at<Vec3b>(i, min) = Vec3b(0, 255, 0);
            min = routeArray[i * cols + min];
            tempImage.at<Vec3b>(i, 0) = Vec3b(0, 0, 255);
        }
    }
    if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
        tempImage = tempImage.t();
    }
    imshow("seam", tempImage);
    waitKey(0);
}

BoundarySegment::BoundarySegment(int _begin, int _end, Direction _direction) :begin(_begin), end(_end), direction(_direction) { }

void BoundarySegment::print()
{
    cout << direction << ' ' << begin << ' ' << end << endl;
}
