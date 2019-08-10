#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "SeamCarving.h"
#include "GlobalWraping.h"
using namespace std;
using namespace cv;


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
#ifndef SKIP_LOCAL



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
        //sc.showCost(bs);

        sc.insertSeam(bs);

        bs = sc.getLongestBoundary();
    }



#ifdef USE_GRAY
    imshow("result", sc.expandGrayImage);
    imwrite("result.png", sc.expandGrayImage);
#endif // USE_GRAY
#ifdef USE_RGB
    imshow("result", sc.expandImage);
    imwrite("result.png", sc.expandImage);
#endif // USE_RGB
#ifdef SHOW_COST
    imwrite("seam.png", sc.seamImage);
#endif // SHOW_COST
    sc.placeMesh();

#ifdef SAVE_MESH
    ofstream out("mesh.txt");
    out << sc.meshCols << ' ' << sc.meshRows << endl;
    for (int i = 0; i <= sc.meshRows; i++) {
        for (int j = 0; j <= sc.meshCols; j++) {
            out << sc.mesh[i][j].x << ' ' << sc.mesh[i][j].y << ' ';
        }
    }
    return 0;
#endif // SAVE_MESH

    
    
    
    GlobalWraping gw(image, mask, sc.mesh, sc.meshRows, sc.meshCols);
#else
    ifstream in("mesh.txt");
    int meshRows, meshCols;
    in >> meshCols >> meshRows;
    Point** mesh = new Point * [meshRows + 1];
    for (int i = 0; i <= meshRows; i++) {
        mesh[i] = new Point[meshCols + 1];
        for (int j = 0; j <= meshCols; j++) {
            in >> mesh[i][j].x >> mesh[i][j].y;
        }
    }
    GlobalWraping gw(image, mask, mesh, meshRows, meshCols);

#endif // !SKIP_LOCAL


    gw.drawMesh(gw.meshVertex);
    gw.calcMeshToVertex();
    gw.calcBoundaryEnergy();
    gw.calcMeshShapeEnergy();
    gw.detectLineSegment();
    gw.calcCost(gw.meshVertex);
    gw.test(gw.meshVertex, "mesh");
    gw.calcMeshLineEnergy();
    for (int i = 0; i < 10; i++) {
        cout << "iter: " << i + 1 << endl;
        gw.updateV();
        //gw.test(gw.newMeshVertex, "updateV");
        gw.calcCost(gw.newMeshVertex);
        gw.updateTheta();
        gw.calcMeshLineEnergy();
        //gw.test(gw.newMeshVertex, "updateV");

        gw.calcLineCost(gw.newMeshVertex);
        //gw.calcCost(gw.newMeshVertex);
        gw.drawMesh(gw.newMeshVertex);
    }
    waitKey(0);

    return 0;
}
