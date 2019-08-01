#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "SeamCarving.h"
using namespace std;
using namespace cv;
int main(int argc, const char* argv[])
{
	//if (argc == 1) {
	//	std::cout << "Please enter the file name." << endl;
	//	return 1;
	//}
	string maskFilename = "pano-mask-4.png";
	string imageFilename = "pano-4.png";
	Mat image = imread(imageFilename);
	Mat mask = imread(maskFilename, IMREAD_GRAYSCALE);
	string maskFilename1 = "pano-mask-5.png";
	string imageFilename1 = "pano-5.png";
	Mat image1 = imread(imageFilename1);
	Mat mask1 = imread(maskFilename1, IMREAD_GRAYSCALE);
	//if (image.empty()) {
	//	cout << "Cannot open file." << endl;
	//	return 1;
	//}
	SeamCarving sc(image, mask);
	BoundarySegment bs;
	SeamCarving sc1(image1, mask1);
	BoundarySegment bs1;
	double time = 0, time1 = 0;
	for (int i = 0; i < 140; i++) {
		auto start = chrono::system_clock::now();
		bs = sc.getLongestBoundary();
		sc.calcCost(bs);
		sc.insertSeam(bs);
		time += double((chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start)).count());
		start = chrono::system_clock::now();
		bs1 = sc1.getLongestBoundary();
		sc1.calcCost(bs1);
		sc1.insertSeam(bs1);
		time1 += double((chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start)).count());

	}
	//bs = sc.getLongestBoundary();
	//sc.calcCost(bs);
	//sc.showCost(bs);
	//sc.getLongestBoundary();
	cout << time << endl << time1 << endl;
	imshow("result", sc.grayImage);
	imshow("result1", sc1.grayImage);
	waitKey(0);
	return 0;
}
