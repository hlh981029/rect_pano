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
	string maskFilename = "pano-mask-2.jpg";
	string imageFilename = "pano-2.jpg";
	Mat image = imread(imageFilename);
	Mat mask = imread(maskFilename, IMREAD_GRAYSCALE);
	//if (image.empty()) {
	//	cout << "Cannot open file." << endl;
	//	return 1;
	//}
	SeamCarving sc(image, mask);
	//sc.calcCost();
	//sc.showCost();
	sc.resize(0, 0);
	waitKey(0);
	return 0;
}