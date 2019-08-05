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
		expandMaskArray = expandMaskImage.data;
		expandGrayArray = (float*)expandGrayImage.data;
		image = image.t();
		swap(cols, rows);
		directionMask = Vertical;
	}
}

void SeamCarving::calcCost(BoundarySegment boundarySegment)
{
	// Transpose if need horizontal seam
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

	// Initialize
	M.setTo(0);
	route.setTo(-1);
	int begin = boundarySegment.begin, end = boundarySegment.end;
	double tempLeftCost, tempUpCost, tempRightCost;
	int left, right, upLeft, upRight, up, middle, leftBorder, rightBorder, upLeftBorder, upRightBorder;
	bool hasUp, hasLeft, hasRight;
	int minDirection;
	double minCost;
	// Dynamic program
	for (int i = begin; i <= end; i++) {

		// Top row of image
		if (i == 0) {
			continue;
		}
		// Find available pixels after removing seams
		leftBorder = upLeftBorder = 0;
		rightBorder = upRightBorder = cols - 1;
		for (int j = 0; j < cols; j++) {
			neighborIndexArray[j][0] = neighborIndexArray[j][1] = j;
		}
		for (int j = 0; j < cols; j++) {
			if (neighborIndexArray[j][0] != -1 && (expandMaskArray[i * cols + neighborIndexArray[j][0]] & (directionMask & (Left | Top))) != 0) {
				for (int k = j; k > 0; k--) {
					neighborIndexArray[k][0] = neighborIndexArray[k - 1][0];
					if (neighborIndexArray[k - 1][0] == -1) {
						break;
					}
				}
				neighborIndexArray[0][0] = -1;
				leftBorder++;
			}
			if (neighborIndexArray[j][1] != -1 && (expandMaskArray[(i - 1) * cols + neighborIndexArray[j][1]] & (directionMask & (Left | Top))) != 0) {
				for (int k = j; k > 0; k--) {
					neighborIndexArray[k][1] = neighborIndexArray[k - 1][1];
					if (neighborIndexArray[k - 1][1] == -1) {
						break;
					}
				}
				neighborIndexArray[0][1] = -1;
				upLeftBorder++;
			}
		}
		for (int j = cols - 1; j >= 0; j--) {
			if (neighborIndexArray[j][0] != -1 && (expandMaskArray[i * cols + neighborIndexArray[j][0]] & (directionMask & (Right | Bottom))) != 0) {
				for (int k = j; k < cols - 1; k++) {
					neighborIndexArray[k][0] = neighborIndexArray[k + 1][0];
					if (neighborIndexArray[k + 1][0] == -1) {
						break;
					}
				}
				neighborIndexArray[cols - 1][0] = -1;
				rightBorder--;
			}
			if (neighborIndexArray[j][1] != -1 && (expandMaskArray[(i - 1) * cols + neighborIndexArray[j][1]] & (directionMask & (Right | Bottom))) != 0) {
				for (int k = j; k < cols - 1; k++) {
					neighborIndexArray[k][1] = neighborIndexArray[k + 1][1];
					if (neighborIndexArray[k + 1][1] == -1) {
						break;
					}
				}
				neighborIndexArray[cols - 1][1] = -1;
				upRightBorder--;
			}
		}
		assert((leftBorder <= rightBorder) && (upLeftBorder <= upRightBorder));
		for (int j = 0; j < cols; j++) {
			assert(neighborIndexArray[j][0] == -1 || (expandMaskArray[i * cols + neighborIndexArray[j][0]] & directionMask) == 0);
			assert(neighborIndexArray[j][1] == -1 || (expandMaskArray[(i - 1) * cols + neighborIndexArray[j][1]] & directionMask) == 0);
		}

		// Calculate cost
		for (int j = 0; j < cols; j++) {
			if (neighborIndexArray[j][0] == -1) {
				continue;
			}
			hasUp = hasLeft = hasRight = true;
			middle = neighborIndexArray[j][0];
			if (j == 0 || middle == 0) {
				hasLeft = false;
				left = neighborIndexArray[j][0];
				if (neighborIndexArray[j][1] == -1) {
					hasUp = false;
					up = neighborIndexArray[j][0];
					if (neighborIndexArray[j + 1][1] == -1) {
						hasRight = false;
					}
					else {
						upRight = neighborIndexArray[j + 1][1];
					}
				}
				else {
					up = neighborIndexArray[j][1];
					if (neighborIndexArray[j + 1][1] == -1) {
						hasRight = false;
					}
					else {
						upRight = neighborIndexArray[j + 1][1];
					}
				}
				if (neighborIndexArray[j + 1][0] == -1) {
					assert(false);
					right = neighborIndexArray[leftBorder][0];
				}
				else {
					right = neighborIndexArray[j + 1][0];
				}
			}
			else if (j == cols - 1 || middle == cols - 1) {
				hasRight = false;
				right = neighborIndexArray[j][0];
				if (neighborIndexArray[j][1] == -1) {
					hasUp = false;
					up = neighborIndexArray[j][0];
					if (neighborIndexArray[j - 1][1] == -1) {
						hasLeft = false;
					}
					else {
						upLeft = neighborIndexArray[j - 1][1];
					}
				}
				else {
					up = neighborIndexArray[j][1];
					if (neighborIndexArray[j - 1][1] == -1) {
						hasLeft = false;
					}
					else {
						upLeft = neighborIndexArray[j - 1][1];
					}
				}
				if (neighborIndexArray[j - 1][0] == -1) {
					assert(false);
					left = neighborIndexArray[rightBorder][0];
				}
				else {
					left = neighborIndexArray[j - 1][0];
				}
			}
			else {
				if (neighborIndexArray[j - 1][0] == -1) {
					left = neighborIndexArray[j][0];
				}
				else {
					left = neighborIndexArray[j - 1][0];
				}
				if (neighborIndexArray[j + 1][0] == -1) {
					right = neighborIndexArray[j][0];
				}
				else {
					right = neighborIndexArray[j + 1][0];
				}
				if (neighborIndexArray[j][1] == -1) {
					hasUp = false;
					assert(!(neighborIndexArray[j - 1][1] != -1 && neighborIndexArray[j + 1][1] != -1));
					if (neighborIndexArray[j - 1][1] == -1) {
						hasLeft = false;
					}
					else {
						assert(neighborIndexArray[j + 1][1] == -1);
						up = neighborIndexArray[j][0];
						upLeft = neighborIndexArray[j - 1][1];
					}
					if (neighborIndexArray[j + 1][1] == -1) {
						hasRight = false;
					}
					else {
						assert(neighborIndexArray[j - 1][1] == -1);
						up = neighborIndexArray[j][0];
						upRight = neighborIndexArray[j + 1][1];
					}
				}
				else {
					up = neighborIndexArray[j][1];
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
			}

			if (!(hasUp || hasRight || hasLeft)) {
				mArray[i * cols + middle] = 10e9;
				continue;
			}
			minDirection = -1;
			temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left);
			tempUpCost = 0.0 + abs(temp[0]) + abs(temp[1]) + abs(temp[2]);

			if (hasUp) {
				if (hasLeft) {
					assert(upLeft < cols && upLeft >= 0);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, left);
					tempLeftCost = tempUpCost + mArray[(i - 1) * cols + upLeft] + abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					minDirection = upLeft;
					minCost = tempLeftCost;
				}
				if (hasRight) {
					assert(upRight < cols && upRight >= 0);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, right);
					tempRightCost = tempUpCost + mArray[(i - 1) * cols + upRight] + abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					if (minDirection == -1 || tempRightCost < minCost) {
						minDirection = upRight;
						minCost = tempRightCost;
					}
				}
				tempUpCost += mArray[(i - 1) * cols + up];
				if (minDirection == -1 || tempUpCost < minCost) {
					minDirection = up;
					minCost = tempUpCost;
				}
			}
			else {
				if (hasLeft) {
					assert(upLeft < cols && upLeft >= 0);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, left);
					tempLeftCost = tempUpCost + mArray[(i - 1) * cols + upLeft] + abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					minDirection = upLeft;
					minCost = tempLeftCost;
				}
				if (hasRight) {
					assert(upRight < cols && upRight >= 0);
					temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, right);
					tempRightCost = tempUpCost + mArray[(i - 1) * cols + upRight] + abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
					if (minDirection == -1 || tempRightCost < minCost) {
						minDirection = upRight;
						minCost = tempRightCost;
					}
				}
			}
			assert(minDirection != -1);
			assert((expandMaskArray[(i - 1) * cols + minDirection] & directionMask) == 0);
			if (i != begin && i != begin + 1 && routeArray[(i - 1) * cols + minDirection] == -1) {
				routeArray[i * cols + middle] = -1;
				mArray[i * cols + middle] = 10e9;
			}
			else {
				mArray[i * cols + middle] = minCost;
				routeArray[i * cols + middle] = minDirection;
			}

			//assert(i == 1 || i == begin || routeArray[i * cols + middle] == -1 || routeArray[(i - 1) * cols + minDirection] != -1);
			if (expandMaskArray[i * cols + middle] == 0) {
				mArray[i * cols + middle] += MAX_COST;
			}
		}
		//		for (int j = 0; j < cols; j++) {
		//			if ((expandMaskArray[i * cols + j] & directionMask) != 0) {
		//				continue;
		//			}
		//			upLeft = neighborIndexArray[j][2];
		//			upRight = neighborIndexArray[j][3];
		//			up = neighborIndexArray[j][4];
		//			left = neighborIndexArray[j][0];
		//			right = neighborIndexArray[j][1];
		//
		//			assert(up >= 0 && up < cols);
		//			if (left < 0) {
		//#ifdef USE_RGB
		//				temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left + 1);
		//				tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
		//				temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, right);
		//				tempRightCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
		//#endif // USE_RGB
		//#ifdef USE_GRAY
		//				tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
		//				tempRightCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
		//#endif // USE_GRAY
		//				tempUpCost += mArray[(i - 1) * cols + up];
		//				tempRightCost += mArray[(i - 1) * cols + upRight];
		//				if (tempUpCost < tempRightCost) {
		//					mArray[i * cols + j] = tempUpCost;
		//					routeArray[i * cols + j] = up;
		//				}
		//				else {
		//					mArray[i * cols + j] = tempRightCost;
		//					routeArray[i * cols + j] = upRight;
		//				}
		//
		//			}
		//			else if (right >= cols) {
		//#ifdef USE_RGB
		//				temp = expandImage.at<Vec3b>(i, right - 1) - expandImage.at<Vec3b>(i, left);
		//				tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
		//				temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, left);
		//				tempLeftCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
		//#endif // USE_RGB
		//#ifdef USE_GRAY
		//				tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
		//				tempLeftCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
		//#endif // USE_GRAY
		//				tempUpCost += mArray[(i - 1) * cols + up];
		//				tempLeftCost += mArray[(i - 1) * cols + upLeft];
		//				if (tempLeftCost < tempUpCost) {
		//					mArray[i * cols + j] = tempLeftCost;
		//					routeArray[i * cols + j] = upLeft;
		//				}
		//				else {
		//					mArray[i * cols + j] = tempUpCost;
		//					routeArray[i * cols + j] = up;
		//				}
		//
		//			}
		//			else {
		//#ifdef USE_RGB
		//				temp = expandImage.at<Vec3b>(i, right) - expandImage.at<Vec3b>(i, left);
		//				tempUpCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
		//				temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, left);
		//				tempLeftCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
		//				temp = expandImage.at<Vec3b>(i - 1, up) - expandImage.at<Vec3b>(i, right);
		//				tempRightCost = abs(temp[0]) + abs(temp[1]) + abs(temp[2]);
		//#endif // USE_RGB
		//#ifdef USE_GRAY
		//				tempUpCost = abs(expandGrayArray[i * cols + right] - expandGrayArray[i * cols + left]);
		//				tempLeftCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + left]);
		//				tempRightCost = tempUpCost + abs(expandGrayArray[(i - 1) * cols + up] - expandGrayArray[i * cols + right]);
		//#endif // USE_GRAY
		//				tempUpCost += mArray[(i - 1) * cols + up];
		//				tempLeftCost += mArray[(i - 1) * cols + upLeft];
		//				tempRightCost += mArray[(i - 1) * cols + upRight];
		//				if (tempLeftCost < tempUpCost) {
		//					if (tempLeftCost < tempRightCost) {
		//						mArray[i * cols + j] = tempLeftCost;
		//						routeArray[i * cols + j] = upLeft;
		//					}
		//					else {
		//						mArray[i * cols + j] = tempRightCost;
		//						routeArray[i * cols + j] = upRight;
		//					}
		//				}
		//				else if (tempUpCost < tempRightCost) {
		//					mArray[i * cols + j] = tempUpCost;
		//					routeArray[i * cols + j] = up;
		//				}
		//				else {
		//					mArray[i * cols + j] = tempRightCost;
		//					routeArray[i * cols + j] = upRight;
		//				}
		//
		//			}
		//			if (expandMaskArray[i * cols + j] == 0) {
		//				mArray[i * cols + j] += MAX_COST;
		//			}
		//			assert((expandMaskArray[(i - 1) * cols + routeArray[i * cols + j]] & directionMask) == 0);
		//			assert(routeArray[i * cols + j] >= 0 && routeArray[i * cols + j] < cols);
		//		}
	}
	for (int i = 0; i < cols; i++) {
		if ((expandMaskArray[end * cols + i] & directionMask) != 0) {
			mArray[end * cols + i] = 1e9;
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
