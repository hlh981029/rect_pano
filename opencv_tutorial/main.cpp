#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "SeamCarving.h"
using namespace std;
using namespace cv;

void resetBorder(Mat& image);
int main(int argc, const char* argv[])
{
    //if (argc == 1) {
    //	std::cout << "Please enter the file name." << endl;
    //	return 1;
    //}
    string maskFilename = "pano-mask-test.png";
    string imageFilename = "pano-test.png";
    Mat image = imread(imageFilename);
    Mat mask = imread(maskFilename, IMREAD_GRAYSCALE);
    SeamCarving sc(image, mask);
    BoundarySegment bs = sc.getLongestBoundary();
    //if (image.empty()) {
    //	cout << "Cannot open file." << endl;
    //	return 1;
    //}

    //string maskFilename1 = "pano-mask-5.png";
    //string imageFilename1 = "pano-5.png";
    //Mat image1 = imread(imageFilename1);
    //Mat mask1 = imread(maskFilename1, IMREAD_GRAYSCALE);
    //SeamCarving sc1(image1, mask1);
    //BoundarySegment bs1;
    //double time = 0, time1 = 0;
    //for (int i = 0; i < 100; i++) {
    //	auto start = chrono::system_clock::now();
    //	bs = sc.getLongestBoundary();
    //	sc.calcCost(bs);
    //	sc.insertSeam(bs);
    //	time += double((chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start)).count());
    //	start = chrono::system_clock::now();
    //	bs1 = sc1.getLongestBoundary();
    //	sc1.calcCost(bs1);
    //	sc1.insertSeam(bs1);
    //	time1 += double((chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start)).count());
    //}
    //cout << time << endl << time1 << endl;
    //imshow("result1", sc1.grayImage);

    //bs = sc.getLongestBoundary();
    //sc.calcCost(bs);
    //sc.showCost(bs);
    //sc.insertSeam(bs);
    //sc.getLongestBoundary();


    //namedWindow("seam", WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    while (bs.direction != None) {
        bs.print();
        sc.calcCost(bs);
        sc.showCost(bs);

        sc.insertSeam(bs);

        bs = sc.getLongestBoundary();
    }
    imshow("result", sc.expandGrayImage);
    Mat writeExpandGrayImage, writeGrayImage;
    sc.expandGrayImage.convertTo(writeExpandGrayImage, CV_8U, 255, 0.5);
    sc.grayImage.convertTo(writeGrayImage, CV_8U, 255, 0.5);
    imwrite("result.png", writeExpandGrayImage);
    imwrite("gray.png", writeGrayImage);
    imwrite("seam.png", image);
    waitKey(0);

    return 0;
}
