#include "SeamCarving.h"
const int MAX_COST = 1e8;

SeamCarving::SeamCarving(Mat& _image, Mat& _mask) :image(_image), mask(_mask)
{
	Mat bgrImage;
	image.convertTo(bgrImage, CV_32FC3, 1.0 / 255);
	rows = image.rows;
	cols = image.cols;
	maxLen = max(cols, rows);
	cvtColor(bgrImage, grayImage, COLOR_BGR2GRAY);
	leftCost.create(maxLen, maxLen, CV_32F);
	upCost.create(maxLen, maxLen, CV_32F);
	rightCost.create(maxLen, maxLen, CV_32F);
	M.create(maxLen, maxLen, CV_32F);
	M.setTo(0);
	route.create(maxLen, maxLen, CV_32S);
	maskArray = mask.data;
	grayImageArray = (float*)grayImage.data;
	leftCostArray = (float*)leftCost.data;
	upCostArray = (float*)upCost.data;
	rightCostArray = (float*)rightCost.data;
	mArray = (float*)M.data;
	routeArray = (int*)route.data;
}

BoundarySegment SeamCarving::getLongestBoundary()
{
	int start, end, length, maxLength = 0, maxStart = 0, maxEnd = 0;
	Direction maxDirection = Direction::None;
	uchar* boundaryMask;
	// Top
	start = end = length = 0;
	boundaryMask = maskArray;
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
	boundaryMask = maskArray + (rows - 1) * cols;
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
	boundaryMask = maskArray;
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
	boundaryMask = maskArray + cols - 1;
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
			for (int j = cols - 1; j > min; j--) {
				grayImageArray[i * cols + j] = grayImageArray[i * cols + j - 1];
				maskArray[i * cols + j] = maskArray[i * cols + j - 1];
			}
			min = routeArray[i * cols + min];
		}
	}
	else {
		for (int i = end; i >= begin; i--) {
			for (int j = 0; j < min; j++) {
				grayImageArray[i * cols + j] = grayImageArray[i * cols + j + 1];
				maskArray[i * cols + j] = maskArray[i * cols + j + 1];
			}
			min = routeArray[i * cols + min];
		}
	}
	if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
		mask = tMask.t();
		grayImage = tGrayImage.t();
		maskArray = mask.data;
		grayImageArray = (float*)grayImage.data;
		swap(cols, rows);
	}
}

void SeamCarving::calcCost(BoundarySegment boundarySegment)
{
	if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
		tMask = mask.t();
		tGrayImage = grayImage.t();
		maskArray = tMask.data;
		grayImageArray = (float*)tGrayImage.data;
		swap(cols, rows);
	}
	M.setTo(0);
	int begin = boundarySegment.begin, end = boundarySegment.end;
	float temp1, temp2, temp3;
#pragma omp parallel for
	for (int i = begin; i < end + 1; i++) {
		for (int j = 0; j < cols; j++) {
			if (maskArray[i * cols + j] == 0) {
				upCostArray[i * cols + j] = leftCostArray[i * cols + j] = rightCostArray[i * cols + j] = MAX_COST;
			}
			else {
				temp1 = abs(grayImageArray[i * cols + (j + 1 < cols ? j + 1 : j)] - grayImageArray[i * cols + (j - 1 >= 0 ? j - 1 : j)]);
				upCostArray[i * cols + j] = temp1;
				leftCostArray[i * cols + j] = temp1 + abs(grayImageArray[(i - 1 >= 0 ? i - 1 : i) * cols + j] - grayImageArray[i * cols + (j - 1 >= 0 ? j - 1 : j)]);
				rightCostArray[i * cols + j] = temp1 + abs(grayImageArray[(i - 1 >= 0 ? i - 1 : i) * cols + j] - grayImageArray[i * cols + (j + 1 < cols ? j + 1 : j)]);
			}
		}
	}
	for (int i = begin; i < end + 1; i++) {
		if (i == 0) {
			continue;
		}
#pragma omp parallel for
		for (int j = 0; j < cols; j++) {
			if (maskArray[i * cols + j] == 0) {
				mArray[i * cols + j] = MAX_COST;
				routeArray[i * cols + j] = -1;
			}
			else {
				temp1 = (j - 1 >= 0) ? mArray[(i - 1) * cols + j - 1] + leftCostArray[i * cols + j] : MAX_COST;
				temp2 = mArray[(i - 1) * cols + j] + upCostArray[i * cols + j];
				temp3 = (j + 1 < cols) ? mArray[(i - 1) * cols + j + 1] + rightCostArray[i * cols + j] : MAX_COST;
				if (temp1 < temp2) {
					if (temp1 < temp3) {
						mArray[i * cols + j] = temp1;
						routeArray[i * cols + j] = j - 1;
					}
					else {
						mArray[i * cols + j] = temp3;
						routeArray[i * cols + j] = j + 1;
					}
				}
				else if (temp2 < temp3) {
					mArray[i * cols + j] = temp2;
					routeArray[i * cols + j] = j;
				}
				else {
					mArray[i * cols + j] = temp3;
					routeArray[i * cols + j] = j + 1;
				}
			}
		}
	}
}

void SeamCarving::showCost(BoundarySegment boundarySegment)
{
	int begin = boundarySegment.begin, end = boundarySegment.end;
	Mat costImage;
	//normalize(leftCost, costImage, 1, 0.0, NORM_MINMAX);
	//imshow("leftCost", costImage);
	//normalize(upCost, costImage, 1, 0.0, NORM_MINMAX);
	//imshow("upCost", costImage);
	//normalize(rightCost, costImage, 1, 0.0, NORM_MINMAX);
	//imshow("rightCost", costImage);
	//normalize(M, costImage, 1, 0.0, NORM_MINMAX);
	//imshow("cost image 0", costImage);
	int min = 0;
	if (boundarySegment.direction == Left || boundarySegment.direction == Right) {
		double temp = mArray[end * cols];
		for (int i = 1; i < cols; i++) {
			if (mArray[end * cols + i] < temp) {
				min = i;
				temp = mArray[end * cols + i];
			}
		}
		mArray[end * cols + min] = 1e9;
		for (int i = end; i > begin; i--) {
			min = routeArray[i * cols + min];
			mArray[(i - 1) * cols + min] = 1e9;
		}
	}
	normalize(M, costImage, 1, 0.0, NORM_MINMAX);
	imshow("cost image 1", costImage);
}

BoundarySegment::BoundarySegment(int _begin, int _end, Direction _direction) :begin(_begin), end(_end), direction(_direction) { }

void BoundarySegment::print()
{
	cout << direction << ' ' << begin << ' ' << end << endl;
}
