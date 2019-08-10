#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <GL/glut.h>
#include "SeamCarving.h"
#include "GlobalWraping.h"
using namespace std;



cv::Mat image;
int glMeshRows, glMeshCols;
Coordinate** meshVertex;
Coordinate** newMeshVertex;
GLuint texGround;

void display();
GLuint matToTexture(cv::Mat mat, GLenum minFilter = GL_LINEAR,
    GLenum magFilter = GL_LINEAR, GLenum wrapFilter = GL_REPEAT);

int main(int argc, char** argv)
{
    //if (argc == 1) {
    //	std::cout << "Please enter the file name." << endl;
    //	return 1;
    //}
    string maskFilename = "pano-mask-test.png";
    string imageFilename = "pano-test.png";
    image = cv::imread(imageFilename);
    cv::Mat mask = cv::imread(maskFilename, cv::IMREAD_GRAYSCALE);
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
    int tempMeshRows, tempMeshCols;
    in >> tempMeshCols >> tempMeshRows;
    cv::Point** mesh = new cv::Point * [tempMeshRows + 1];
    for (int i = 0; i <= tempMeshRows; i++) {
        mesh[i] = new cv::Point[tempMeshCols + 1];
        for (int j = 0; j <= tempMeshCols; j++) {
            in >> mesh[i][j].x >> mesh[i][j].y;
        }
    }
    GlobalWraping gw(image, mask, mesh, tempMeshRows, tempMeshCols);

#endif // !SKIP_LOCAL


    gw.drawMesh(gw.meshVertex);
    gw.calcMeshToVertex();
    gw.calcBoundaryEnergy();
    gw.calcMeshShapeEnergy();
    gw.detectLineSegment();
    gw.calcCost(gw.meshVertex);
    //gw.test(gw.meshVertex, "mesh");
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
        //gw.drawMesh(gw.newMeshVertex);
    }
    //cv::waitKey(0);

    glMeshCols = gw.meshCols;
    glMeshRows = gw.meshRows;
    meshVertex = new Coordinate * [glMeshRows + 1];
    newMeshVertex = new Coordinate * [glMeshRows + 1];
    for (int i = 0; i <= glMeshRows; i++) {
        meshVertex[i] = new Coordinate[glMeshCols + 1];
        newMeshVertex[i] = new Coordinate[glMeshCols + 1];
        for (int j = 0; j <= glMeshCols; j++) {
            meshVertex[i][j].col = gw.meshVertex[i][j].col;
            meshVertex[i][j].row = gw.meshVertex[i][j].row;
            newMeshVertex[i][j].col = gw.newMeshVertex[i][j].col;
            newMeshVertex[i][j].row = gw.newMeshVertex[i][j].row;
            Coordinate& coord = newMeshVertex[i][j];
            Coordinate& localcoord = meshVertex[i][j];
            coord.row /= image.rows;
            coord.col /= image.cols;
            coord.row -= 0.5;
            coord.col -= 0.5;
            coord.row *= 2;
            coord.col *= 2;
            coord.row = median(coord.row, -1, 1);
            coord.col = median(coord.col, -1, 1);
            localcoord.row /= image.rows;
            localcoord.col /= image.cols;
            localcoord.row = median(localcoord.row, 0, 1);
            localcoord.col = median(localcoord.col, 0, 1);
        }
    }
    //glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(image.cols, image.rows);
    glutCreateWindow("rect_pano");
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);    // 启用纹理
    texGround = matToTexture(image);
    glutDisplayFunc(&display);   //注册函数 
    glutMainLoop(); //循环调用
    return 0;
}
void display()
{
    glLoadIdentity();
    // 清除屏幕
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texGround);


    for (int row = 0; row < glMeshRows; row++) {
        for (int col = 0; col < glMeshCols; col++) {
            Coordinate local_left_top = meshVertex[row][col];
            Coordinate local_right_top = meshVertex[row][col + 1];
            Coordinate local_left_bottom = meshVertex[row + 1][col];
            Coordinate local_right_bottom = meshVertex[row + 1][col + 1];


            Coordinate global_left_top = newMeshVertex[row][col];
            Coordinate global_right_top = newMeshVertex[row][col + 1];
            Coordinate global_left_bottom = newMeshVertex[row + 1][col];
            Coordinate global_right_bottom = newMeshVertex[row + 1][col + 1];

            glBegin(GL_QUADS);
            glTexCoord2d(local_right_top.col, local_right_top.row); glVertex3d(global_right_top.col, -1 * global_right_top.row, 0.0f);
            glTexCoord2d(local_right_bottom.col, local_right_bottom.row); glVertex3d(global_right_bottom.col, -1 * global_right_bottom.row, 0.0f);
            glTexCoord2d(local_left_bottom.col, local_left_bottom.row);	glVertex3d(global_left_bottom.col, -1 * global_left_bottom.row, 0.0f);
            glTexCoord2d(local_left_top.col, local_left_top.row); glVertex3d(global_left_top.col, -1 * global_left_top.row, 0.0f);
            glEnd();

        }
    }
    glutSwapBuffers();

}


GLuint matToTexture(cv::Mat mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter)
{
    //cv::flip(mat, mat, 0);
    // Generate a number for our textureID's unique handle
    GLuint textureID;
    glGenTextures(1, &textureID);

    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Catch silly-mistake texture interpolation method for magnification
    if (magFilter == GL_LINEAR_MIPMAP_LINEAR ||
        magFilter == GL_LINEAR_MIPMAP_NEAREST ||
        magFilter == GL_NEAREST_MIPMAP_LINEAR ||
        magFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
        //cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << endl;
        magFilter = GL_LINEAR;
    }

    // Set texture interpolation methods for minification and magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

    // Set incoming texture format to:
    // GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
    // GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
    // Work out other mappings as required ( there's a list in comments in main() )
    GLenum inputColourFormat = GL_BGR_EXT;
    if (mat.channels() == 1)
    {
        inputColourFormat = GL_LUMINANCE;
    }

    // Create the texture
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
        0,                 // Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGB,            // Internal colour format to convert to
        mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
        mat.rows,          // Image height i.e. 480 for Kinect in standard mode
        0,                 // Border width in pixels (can either be 1 or 0)
        inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE,  // Image data type
        mat.data);        // The actual image data itself

                           // If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher

    return textureID;
}

