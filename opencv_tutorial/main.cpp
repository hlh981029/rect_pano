
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <GL/glut.h>
#include "SeamCarving.h"
#include "GlobalWarping.h"
#include "util.h"
using namespace std;



cv::Mat glImage;
int glMeshRows, glMeshCols;
Coordinate** meshVertex;
Coordinate** newMeshVertex;
Coordinate** meshVertex1;
Coordinate** newMeshVertex1;
GLuint texGround;
GLbyte* colorArr;
double scaleRatio, scaleX, scaleY;
string imageName;
void display();
void keyboard(unsigned char key, int x, int y);
void updateGlobalVariable(GlobalWarping& gw, double ratio);
void saveImage();

GLuint matToTexture(cv::Mat mat, GLenum minFilter = GL_LINEAR, GLenum magFilter = GL_LINEAR, GLenum wrapFilter = GL_REPEAT);

int main(int argc, char** argv)
{
    if (argc == 1) {
    	std::cout << "Please enter the file name." << endl;
    	return 1;
    }
    imageName = argv[1];
    string imageFilename = imageName+"_input.jpg";
    string maskFilename = imageName+"_input_mask.jpg";
    cv::Mat image = cv::imread(imageFilename);
    cv::Mat mask = cv::imread(maskFilename, cv::IMREAD_GRAYSCALE);

    scaleRatio = sqrt(WRAPING_RESOLUTION / (image.cols * image.rows));
    cv::Mat scaledImage, scaledMask;
    cv::resize(image, scaledImage, cv::Size(), scaleRatio, scaleRatio);
    cv::resize(mask, scaledMask, cv::Size(), scaleRatio, scaleRatio);

    image.copyTo(glImage);

    auto start = chrono::system_clock::now();
    auto start2 = chrono::system_clock::now();
    SeamCarving sc(scaledImage, scaledMask);
    BoundarySegment bs = sc.getLongestBoundary();
    while (bs.direction != None) {
        //bs.print();
        sc.calcCost(bs);
        //sc.showCost(bs);
        sc.insertSeam(bs);
        bs = sc.getLongestBoundary();
    }

#ifdef USE_GRAY
    //imshow("result", sc.expandGrayImage);
    //imwrite(imageName+"result.jpg", sc.expandGrayImage);
#endif // USE_GRAY
#ifdef USE_RGB
    imshow("result", sc.expandImage);
    imwrite("result.jpg", sc.expandImage);
#endif // USE_RGB
#ifdef SHOW_COST
    imwrite(imageName+"_seam.jpg", sc.seamImage);
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
    double time = double((chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start)).count());
    cout << "localWraping: " << time << "ms" << endl;
    start = chrono::system_clock::now();
    GlobalWarping gw(scaledImage, scaledMask, sc.mesh, sc.meshRows, sc.meshCols);
    gw.imageName = imageName;
    gw.calcMeshToVertex();
    gw.calcBoundaryEnergy();
    gw.calcMeshShapeEnergy();
    gw.detectLineSegment();
    //gw.drawMesh(gw.meshVertex);
    //gw.calcCost(gw.meshVertex);
    gw.calcMeshLineEnergy();
    for (int i = 0; i < 10; i++) {
        gw.updateV();
        //gw.drawMesh(gw.newMeshVertex);
        //gw.calcCost(gw.newMeshVertex);
        gw.updateTheta();
        gw.calcMeshLineEnergy();
        //gw.calcLineCost(gw.newMeshVertex);
    }
    time = double((chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start)).count());
    cout << "globalWraping: " << time << "ms" << endl;

    updateGlobalVariable(gw, scaleRatio);
    //glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(1, 1);
    glutInitWindowSize(glImage.cols, glImage.rows);
    glutCreateWindow("rect_pano");
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);    // 启用纹理

    texGround = matToTexture(glImage);
    glutDisplayFunc(display);   //注册函数 
    glutKeyboardFunc(keyboard);
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
            glTexCoord2d(local_right_top.col, local_right_top.row); glVertex3d(global_right_top.col, -global_right_top.row, 0.0f);
            glTexCoord2d(local_right_bottom.col, local_right_bottom.row); glVertex3d(global_right_bottom.col, -global_right_bottom.row, 0.0f);
            glTexCoord2d(local_left_bottom.col, local_left_bottom.row);	glVertex3d(global_left_bottom.col, -global_left_bottom.row, 0.0f);
            glTexCoord2d(local_left_top.col, local_left_top.row); glVertex3d(global_left_top.col, -global_left_top.row, 0.0f);
            glEnd();

        }
    }
    glutSwapBuffers();

}

void updateGlobalVariable(GlobalWarping& gw, double ratio)
{
    gw.stretch();
    scaleX = gw.scaleX;
    scaleY = gw.scaleY;
    glMeshCols = gw.meshCols;
    glMeshRows = gw.meshRows;
    meshVertex = new Coordinate * [glMeshRows + 1];
    newMeshVertex = new Coordinate * [glMeshRows + 1];
    meshVertex1 = new Coordinate * [glMeshRows + 1];
    newMeshVertex1 = new Coordinate * [glMeshRows + 1];
    for (int i = 0; i <= glMeshRows; i++) {
        meshVertex[i] = new Coordinate[glMeshCols + 1];
        newMeshVertex[i] = new Coordinate[glMeshCols + 1];
        meshVertex1[i] = new Coordinate[glMeshCols + 1];
        newMeshVertex1[i] = new Coordinate[glMeshCols + 1];
        for (int j = 0; j <= glMeshCols; j++) {
            meshVertex[i][j].col = gw.meshVertex[i][j].col / ratio;
            meshVertex[i][j].row = gw.meshVertex[i][j].row / ratio;
            newMeshVertex[i][j].col = gw.newMeshVertex[i][j].col / ratio;
            newMeshVertex[i][j].row = gw.newMeshVertex[i][j].row / ratio;
            meshVertex1[i][j].col = gw.meshVertex[i][j].col / ratio;
            meshVertex1[i][j].row = gw.meshVertex[i][j].row / ratio;
            newMeshVertex1[i][j].col = gw.newMeshVertex[i][j].col / ratio;
            newMeshVertex1[i][j].row = gw.newMeshVertex[i][j].row / ratio;
            Coordinate& coord = newMeshVertex[i][j];
            Coordinate& localcoord = meshVertex[i][j];
            coord.row = coord.row / glImage.rows * 2.0 - 1;
            coord.col = coord.col / glImage.cols * 2.0 - 1;
            coord.row = median(coord.row, -1, 1);
            coord.col = median(coord.col, -1, 1);
            localcoord.row /= glImage.rows;
            localcoord.col /= glImage.cols;
            localcoord.row = median(localcoord.row, 0, 1);
            localcoord.col = median(localcoord.col, 0, 1);
            assert(coord.row >= -1 && coord.row <= 1);
            assert(coord.col >= -1 && coord.col <= 1);
            assert(localcoord.row >= 0 && localcoord.row <= 1);
            assert(localcoord.col >= 0 && localcoord.col <= 1);
        }
    }
    colorArr = new GLbyte[glImage.cols * glImage.rows *3];
}

GLuint matToTexture(cv::Mat mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
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

void saveImage(){
    GLint viewPort[4] = { 0 };
    glGetIntegerv(GL_VIEWPORT, viewPort);
    cv::Mat img(viewPort[3], viewPort[2], CV_8UC3);
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
    //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
    glReadPixels(viewPort[0], viewPort[1], viewPort[2], viewPort[3], GL_BGR_EXT, GL_UNSIGNED_BYTE, img.data);
    cv::flip(img, img, 0);
    cv::Mat tempImage(glImage.rows, glImage.cols, CV_8UC3);
    cv::Mat stretchImage;
    cv::resize(img, tempImage, tempImage.size(), 0, 0, cv::INTER_CUBIC);
    cv::resize(tempImage, stretchImage, cv::Size(), 1.0/scaleX, 1.0/scaleY, cv::INTER_CUBIC);

    cv::imwrite(imageName+"_result.jpg", tempImage);
    cv::imwrite(imageName + "_result_stretch.jpg", stretchImage);
    for (int i = 0; i < glMeshRows; i++) {
        for (int j = 0; j < glMeshCols; j++) {
            cv::line(tempImage, newMeshVertex1[i][j].toPoint(), newMeshVertex1[i + 1][j].toPoint(), cv::Scalar(0, 255, 0), 2);
            cv::line(tempImage, newMeshVertex1[i][j].toPoint(), newMeshVertex1[i][j + 1].toPoint(), cv::Scalar(0, 255, 0), 2);
        }
        cv::line(tempImage, newMeshVertex1[i][glMeshCols].toPoint(), newMeshVertex1[i + 1][glMeshCols].toPoint(), cv::Scalar(0, 255, 0), 2);
    }
    for (int j = 0; j < glMeshCols; j++) {
        cv::line(tempImage, newMeshVertex1[glMeshRows][j].toPoint(), newMeshVertex1[glMeshRows][j + 1].toPoint(), cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(imageName+"_global_mesh.jpg", tempImage);
    glImage.copyTo(tempImage);
    for (int i = 0; i < glMeshRows; i++) {
        for (int j = 0; j < glMeshCols; j++) {
            cv::line(tempImage, meshVertex1[i][j].toPoint(), meshVertex1[i + 1][j].toPoint(), cv::Scalar(0, 255, 0), 2);
            cv::line(tempImage, meshVertex1[i][j].toPoint(), meshVertex1[i][j + 1].toPoint(), cv::Scalar(0, 255, 0), 2);
        }
        cv::line(tempImage, meshVertex1[i][glMeshCols].toPoint(), meshVertex1[i + 1][glMeshCols].toPoint(), cv::Scalar(0, 255, 0), 2);
    }
    for (int j = 0; j < glMeshCols; j++) {
        cv::line(tempImage, meshVertex1[glMeshRows][j].toPoint(), meshVertex1[glMeshRows][j + 1].toPoint(), cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(imageName+"_local_mesh.jpg", tempImage);
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 's':
    case 'S':
        saveImage();
        break;
    default:
        break;
    }
}