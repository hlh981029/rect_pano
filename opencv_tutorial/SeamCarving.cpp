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
	M.create(maxLen, maxLen, CV_32F);
	route.create(maxLen, maxLen, CV_32S);
	M.setTo(0);
	neighborIndexArray = new int* [maxLen];
	for (int i = 0; i < maxLen; i++) {
		neighborIndexArray[i] = new int[5];
	}
	//imageIndexUsed.setTo(0);
	maskArray = mask.data;
	mArray = (float*)M.data;
	routeArray = (int*)route.data;
	//imageIndexUsedArray = imageIndexUsed.data;
	//displacementIndexArray = (int*)displacementIndex.data;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//displacementIndexArray[i * cols + j] = i * cols + j;
			if (expandMaskArray[i * cols + j] != 255) {
				expandMaskArray[i * cols + j] = 0;
			}
			else {
				expandMaskArray[i * cols + j] = 4;
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
	double temp = 1e9;
	for (int i = 0; i < cols; i++) {
		if ((expandMaskArray[end * cols + i] & directionMask) == 0 && mArray[end * cols + i] < temp) {
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
				expandMaskArray[i * cols + min + 1] = expandMaskArray[i * cols + min] = (expandMaskArray[i * cols + min] | directionMask);
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
				expandMaskArray[i * cols + min - 1] = expandMaskArray[i * cols + min] = (expandMaskArray[i * cols + min] | directionMask);
			}
			min = routeArray[i * cols + min];

		}
	}
	if (boundarySegment.direction == Top || boundarySegment.direction == Bottom) {
		expandImage = expandImage.t();
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
		expandImage = expandImage.t();
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
		if (i == 0) {
			for (int j = 0; j < cols; j++) {
				neighborIndexArray[j][0] = left;
				if ((expandMaskArray[i * cols + j] & directionMask) == 0) {
					left = j;
				}
				neighborIndexArray[cols - 1 - j][1] = right;
				if ((expandMaskArray[i * cols + cols - 1 - j] & directionMask) == 0) {
					right = cols - 1 - j;
				}
			}
			for (int j = 0; j < cols; j++) {
				if ((expandMaskArray[i * cols + j] & directionMask) != 0) {
					continue;
				}
				left = neighborIndexArray[j][0];
				right = neighborIndexArray[j][1];
				if (left < 0) {
#ifdef USE_RGB
						temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left + 1);
						tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
						tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
						mArray[i * cols + j] = tempUpCost;
				}
				else if (right >= cols) {
#ifdef USE_RGB
						temp = expandImage.at<Vec3b>(i, right - 1) - expandImage.at<Vec3b>(i, left);
						tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
						tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
						mArray[i * cols + j] = tempUpCost;
				}
				else {
#ifdef USE_RGB
						temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left);
						tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
						tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
						mArray[i * cols + j] = tempUpCost;
				}
				if (expandMaskArray[i * cols + j] == 0) {
					mArray[i * cols + j] += MAX_COST;
				}
			}
			continue;
		}
		left = upLeft = -1;
		right = upRight = cols;
		for (int j = 0; j < cols; j++) {
			neighborIndexArray[j][0] = left;
			if ((expandMaskArray[i * cols + j] & directionMask) == 0) {
				left = j;
			}
			neighborIndexArray[j][2] = upLeft;
			if ((expandMaskArray[(i-1) * cols + j] & directionMask) == 0) {
				upLeft = j;
			}
			neighborIndexArray[cols-1-j][1] = right;
			if ((expandMaskArray[i * cols + cols - 1 - j] & directionMask) == 0) {
				right = cols - 1 - j;
			}
			neighborIndexArray[cols - 1 - j][3] = upRight;
			if ((expandMaskArray[(i-1) * cols + cols - 1 - j] & directionMask) == 0) {
				upRight = cols - 1 - j;
			}
		}
		for (int j = 0; j < cols; j++) {
			up = j;
			upLeft = neighborIndexArray[j][2];
			upRight = neighborIndexArray[j][3];
			if ((expandMaskArray[(i-1) * cols + j] & directionMask) != 0) {
				upLeft = neighborIndexArray[j][2];
				upRight = neighborIndexArray[j][3];
				if (neighborIndexArray[up][0] == -1) {
					if (upLeft >= 0) {
						up = upLeft;
						upLeft = neighborIndexArray[upLeft][2];
					}
					else {
						up = upRight;
						upRight = neighborIndexArray[upRight][3];
					}
				}
				else if (neighborIndexArray[up][1] == cols) {
					if (upRight < cols) {
						up = upRight;
						upRight = neighborIndexArray[upRight][3];
					}
					else {
						up = upLeft;
						upLeft = neighborIndexArray[upLeft][2];
					}
				}
				else {
					int diffLeft = up - upLeft;
					int diffRight = upRight - up;
					while (diffLeft == diffRight) {
						diffLeft = neighborIndexArray[upLeft][2] - upLeft;
						upLeft = neighborIndexArray[upLeft][2];
						diffRight = neighborIndexArray[upRight][3] - upRight;
						upRight = neighborIndexArray[upRight][3];
					}
					upLeft = neighborIndexArray[up][2];
					upRight = neighborIndexArray[up][3];
					if (diffLeft < diffRight) {
						if (upLeft > 0 && neighborIndexArray[upLeft][2] >= 0) {
							up = upLeft;
							upLeft = neighborIndexArray[upLeft][2];
						}
						else {
							up = upRight;
							upRight = neighborIndexArray[upRight][3];
						}
					}
					else {
						if (upRight < cols - 1 && neighborIndexArray[upRight][3] < cols) {
							up = upRight;
							upRight = neighborIndexArray[upRight][3];
						}
						else {
							up = upLeft;
							upLeft = neighborIndexArray[upLeft][2];
						}
					}
				}
			}
			neighborIndexArray[j][2] = upLeft;
			neighborIndexArray[j][3] = upRight;
			neighborIndexArray[j][4] = up;
		}
		for (int j = 0; j < cols; j++) {
			if ((expandMaskArray[i * cols + j] & directionMask) != 0) {
				continue;
			}
			upLeft = neighborIndexArray[j][2];
			upRight = neighborIndexArray[j][3];
			up = neighborIndexArray[j][4];
			left = neighborIndexArray[j][0];
			right = neighborIndexArray[j][1];

			assert(up >= 0 && up < cols);
			if (left < 0) {
				if (upRight >= cols) {
#ifdef USE_RGB
					temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left+1);
					tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
					tempUpCost += mArray[(i - 1) * cols + up];
					mArray[i * cols + j] = tempUpCost;
					routeArray[i * cols + j] = up;
				}
				else {
#ifdef USE_RGB
					temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left+1);
					tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, right);
					tempRightCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempRightCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
#endif // USE_GRAY
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
			else if (right >= cols) {
				if (upLeft < 0) {
#ifdef USE_RGB
					temp = expandImage.at<Vec3b>(i, right-1) - expandImage.at<Vec3b>(i, left);
					tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
					tempUpCost += mArray[(i - 1) * cols + up];
					mArray[i * cols + j] = tempUpCost;
					routeArray[i * cols + j] = up;
				}
				else {
#ifdef USE_RGB
					temp = expandImage.at<Vec3b>(i, right-1) - expandImage.at<Vec3b>(i, left);
					tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, left);
					tempLeftCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempLeftCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
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
				if (upLeft < 0 && upRight >= cols) {
#ifdef USE_RGB
					temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left);
					tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
					tempUpCost += mArray[(i - 1) * cols + up];
					mArray[i * cols + j] = tempUpCost;
					routeArray[i * cols + j] = up;
				}
				else if (upRight >= cols) {
#ifdef USE_RGB
					temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left);
					tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, left);
					tempLeftCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempLeftCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
#endif // USE_GRAY
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
				else if (upLeft < 0) {
#ifdef USE_RGB
					temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left);
					tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, right);
					tempRightCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempRightCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
#endif // USE_GRAY
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
#ifdef USE_RGB
					temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left);
					tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, left);
					tempLeftCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, right);
					tempRightCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
#endif // USE_RGB
#ifdef USE_GRAY
					tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
					tempLeftCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
					tempRightCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
#endif // USE_GRAY
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
			assert((expandMaskArray[(i - 1) * cols + routeArray[i * cols + j]] & directionMask) == 0);
			assert(routeArray[i * cols + j] >= 0 && routeArray[i * cols + j] < cols);
		}
	}
}

void SeamCarving::showCost(BoundarySegment boundarySegment)
{
	Mat tempImage;
	image.copyTo(tempImage);
	int begin = boundarySegment.begin, end = boundarySegment.end;
	int min = 0;
	double temp = 1e9;
	for (int i = 0; i < cols; i++) {
		if ((expandMaskArray[end * cols + i] & directionMask) == 0 && mArray[end * cols + i] < temp) {
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
