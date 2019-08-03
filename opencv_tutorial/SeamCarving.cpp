#include "SeamCarving.h"
const int MAX_COST = 1e6;

SeamCarving::SeamCarving(Mat& _image, Mat& _mask) :image(_image), mask(_mask)
{
	Mat bgrImage;
	image.convertTo(bgrImage, CV_32FC3, 1.0 / 255);
	rows = image.rows;
	cols = image.cols;
	maxLen = max(cols, rows);
	cvtColor(bgrImage, grayImage, COLOR_BGR2GRAY);


	mask.copyTo(expandMaskImage);
	grayImage.copyTo(expandGrayImage);
	expandGrayArray = (float*)expandGrayImage.data;
	expandMaskArray = expandMaskImage.data;


	//displacementIndex.create(rows, cols, CV_32S);
	//imageIndexUsed.create(rows, cols, CV_8U);
	M.create(maxLen, maxLen, CV_32F);
	route.create(maxLen, maxLen, CV_32S);
	M.setTo(0);
	//imageIndexUsed.setTo(0);
	maskArray = mask.data;
	mArray = (float*)M.data;
	routeArray = (int*)route.data;
	//imageIndexUsedArray = imageIndexUsed.data;
	//displacementIndexArray = (int*)displacementIndex.data;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//displacementIndexArray[i * cols + j] = i * cols + j;
			if (expandMaskArray[i * cols + j] != 0) {
				expandMaskArray[i * cols + j] = 4;
			}
			else {
				expandMaskArray[i * cols + j] = 0;
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
	double temp = mArray[end * cols];

	for (int i = 1; i < cols; i++) {
		if (mArray[end * cols + i] < temp) {
			min = i;
			temp = mArray[end * cols + i];
		}
	}
	if (boundarySegment.direction == Bottom || boundarySegment.direction == Right) {
		for (int i = end; i >= begin; i--) {
			//assert(min < 0 || min >= cols);
			for (int j = cols - 1; j > min; j--) {
				expandGrayArray[i * cols + j] = expandGrayArray[i * cols + j - 1];
				expandMaskArray[i * cols + j] = expandMaskArray[i * cols + j - 1];
				image.at<Vec3b>(i, j) = image.at<Vec3b>(i, j - 1);
			}
			if (expandMaskArray[i * cols + min] != 0) {
				image.at<Vec3b>(i, min) = Vec3b(0,0,255);
				expandMaskArray[i * cols + min + 1] = expandMaskArray[i * cols + min] = expandMaskArray[i * cols + min] | directionMask;
			}
			min = routeArray[i * cols + min];
		}
	}
	else {
		for (int i = end; i >= begin; i--) {
			//assert(min < 0 || min >= cols);
			for (int j = 0; j < min; j++) {
				expandGrayArray[i * cols + j] = expandGrayArray[i * cols + j + 1];
				expandMaskArray[i * cols + j] = expandMaskArray[i * cols + j + 1];
				image.at<Vec3b>(i, j) = image.at<Vec3b>(i, j + 1);
			}
			if (expandMaskArray[i * cols + min] != 0) {
				image.at<Vec3b>(i, min) = Vec3b(0, 0, 255);
				expandMaskArray[i * cols + min - 1] = expandMaskArray[i * cols + min] = expandMaskArray[i * cols + min] | directionMask;
			}
			min = routeArray[i * cols + min];
		}
	}
	if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
		expandMaskImage = expandMaskImage.t();
		expandGrayImage = expandGrayImage.t();
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
		image = image.t();
		expandMaskImage = expandMaskImage.t();
		expandGrayImage = expandGrayImage.t();
		expandMaskArray = expandMaskImage.data;
		expandGrayArray = (float*)expandGrayImage.data;
		swap(cols, rows);
		directionMask = Horizontal;
	}
	M.setTo(0);
	int begin = boundarySegment.begin, end = boundarySegment.end;
	float tempLeftCost, tempUpCost, tempRightCost;
	int left, right, upLeft, upRight, up;
	for (int i = begin; i < end + 1; i++) {
		for (int j = 0; j < cols; j++) {
			if (i == 0) {
				continue;
			}
			upLeft = upRight = up = j;
			left = j - 1;
			right = j + 1;
			while (left >= 0 && (expandMaskArray[i * cols + left] & directionMask) != 0) {
				left--;
			}
			while (right < cols && (expandMaskArray[i * cols + right] & directionMask) != 0) {
				right++;
			}
			while (upLeft >= 0 && (expandMaskArray[(i - 1) * cols + upLeft] & directionMask) != 0) {
				upLeft--;
			}
			while (upRight < cols && (expandMaskArray[(i - 1) * cols + upRight] & directionMask) != 0) {
				upRight++;
			}
			if (upLeft == upRight) {
				upLeft--;
				upRight++;
				while (upLeft >= 0 && (expandMaskArray[(i - 1) * cols + upLeft] & directionMask) != 0) {
					upLeft--;
				}
				while (upRight < cols && (expandMaskArray[(i - 1) * cols + upRight] & directionMask) != 0) {
					upRight++;
				}
			}
			else if ((up - upLeft) < (upRight - up)) {
				up = upLeft;
				upLeft--;
				while (upLeft >= 0 && (expandMaskArray[(i - 1) * cols + upLeft] & directionMask) != 0) {
					upLeft--;
				}
			}
			else {
				up = upRight;
				upRight++;
				while (upRight < cols && (expandMaskArray[(i - 1) * cols + upRight] & directionMask) != 0) {
					upRight++;
				}
			}
			if (left == -1 || expandMaskArray[i * cols + left] == 0) {
				if (upRight == cols || expandMaskArray[(i - 1) * cols + upRight] == 0) {
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempUpCost += mArray[(i - 1) * cols + up];
					mArray[i * cols + j] = tempUpCost;
					routeArray[i * cols + j] = up;
				}
				else {
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempRightCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
					tempUpCost += mArray[(i - 1) * cols + up];
					tempRightCost += mArray[(i - 1) * cols + upRight];
					if (tempUpCost < tempRightCost) {
						mArray[i * cols + j] = tempUpCost;
						routeArray[i * cols + j] = up;
					}
					else {
						mArray[i * cols + j] = tempRightCost;
						routeArray[i * cols + j] = upRight;
					}
				}
			}
			else if (right == cols || expandMaskArray[i * cols + right] == 0) {
				if (upLeft == -1 || expandMaskArray[(i - 1) * cols + upLeft] == 0) {
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempUpCost += mArray[(i - 1) * cols + up];
					mArray[i * cols + j] = tempUpCost;
					routeArray[i * cols + j] = up;
				}
				else {
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempLeftCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
					tempUpCost += mArray[(i - 1) * cols + up];
					tempLeftCost += mArray[(i - 1) * cols + upLeft];
					if (tempLeftCost < tempUpCost) {
						mArray[i * cols + j] = tempLeftCost;
						routeArray[i * cols + j] = upLeft;
					}
					else {
						mArray[i * cols + j] = tempUpCost;
						routeArray[i * cols + j] = up;
					}
				}
			}
			else {
				if (upRight == cols || expandMaskArray[(i - 1) * cols + upRight] == 0) {
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempLeftCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
					tempUpCost += mArray[(i - 1) * cols + up];
					tempLeftCost += mArray[(i - 1) * cols + upLeft];
					if (tempLeftCost < tempUpCost) {
						mArray[i * cols + j] = tempLeftCost;
						routeArray[i * cols + j] = upLeft;
					}
					else {
						mArray[i * cols + j] = tempUpCost;
						routeArray[i * cols + j] = up;
					}
				}
				else if (upLeft == -1 || expandMaskArray[(i - 1) * cols + upLeft] == 0) {
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempRightCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
					tempUpCost += mArray[(i - 1) * cols + up];
					tempRightCost += mArray[(i - 1) * cols + upRight];
					if (tempUpCost < tempRightCost) {
						mArray[i * cols + j] = tempUpCost;
						routeArray[i * cols + j] = up;
					}
					else {
						mArray[i * cols + j] = tempRightCost;
						routeArray[i * cols + j] = upRight;
					}
				}
				else {
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempLeftCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
					tempRightCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
					tempUpCost += mArray[(i - 1) * cols + up];
					tempLeftCost += mArray[(i - 1) * cols + upLeft];
					tempRightCost += mArray[(i - 1) * cols + upRight];
					if (tempLeftCost < tempUpCost) {
						if (tempLeftCost < tempRightCost) {
							mArray[i * cols + j] = tempLeftCost;
							routeArray[i * cols + j] = upLeft;
						}
						else {
							mArray[i * cols + j] = tempRightCost;
							routeArray[i * cols + j] = upRight;
						}
					}
					else if (tempUpCost < tempRightCost) {
						mArray[i * cols + j] = tempUpCost;
						routeArray[i * cols + j] = up;
					}
					else {
						mArray[i * cols + j] = tempRightCost;
						routeArray[i * cols + j] = upRight;
					}
				}
			}
			if (expandMaskArray[i * cols + j] == 0) {
				mArray[i * cols + j] += MAX_COST;
			}
		}
	}
}

void SeamCarving::showCost(BoundarySegment boundarySegment)
{
	int begin = boundarySegment.begin, end = boundarySegment.end;
	Mat costImage(rows, cols, CV_32F);
	//normalize(leftCost, costImage, 1, 0.0, NORM_MINMAX);
	//imshow("leftCost", costImage);
	//normalize(upCost, costImage, 1, 0.0, NORM_MINMAX);
	//imshow("upCost", costImage);
	//normalize(rightCost, costImage, 1, 0.0, NORM_MINMAX);
	//imshow("rightCost", costImage);
	//normalize(M, costImage, 1, 0.0, NORM_MINMAX);
	//imshow("cost image 0", costImage);
	int min = 0;
	double temp = mArray[end * cols];
	for (int i = 1; i < cols; i++) {
		if (mArray[end * cols + i] < temp) {
			min = i;
			temp = mArray[end * cols + i];
		}
	}
	((float*)costImage.data)[end * cols + min] = 1;
	for (int i = end; i > begin; i--) {
		min = routeArray[i * cols + min];
		((float*)costImage.data)[(i - 1) * cols + min] = 1;
	}

	normalize(costImage, costImage, 1, 0.0, NORM_MINMAX);
	imshow("cost image 1", costImage);
}

BoundarySegment::BoundarySegment(int _begin, int _end, Direction _direction) :begin(_begin), end(_end), direction(_direction) { }

void BoundarySegment::print()
{
	cout << direction << ' ' << begin << ' ' << end << endl;
}
