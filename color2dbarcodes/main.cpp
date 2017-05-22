#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "decoding.h"

using namespace std;
using namespace cv;


#define Mpixel(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)]

#define MpixelB(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())]
#define MpixelG(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())+1]
#define MpixelR(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())+2]



int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("needs an image\n");
        exit(0);
    }

    Mat colorImg = imread(argv[1], 1);
    if( colorImg.empty() )
    {
        printf("cannot open the image \n");
        exit(0);
    }

    Mat greyImg;
    cvtColor(colorImg, greyImg, CV_RGB2GRAY);

    medianBlur(greyImg, greyImg, 5);
    GaussianBlur(greyImg, greyImg, cv::Size(5, 5), 3, 3);
    vector<Vec3f> circles;
    cout<<"greyImg.rows/4="<<greyImg.rows/5<<endl;
    HoughCircles(greyImg, circles, CV_HOUGH_GRADIENT, 1, greyImg.rows/5, 180, 100, 100, 500);
    cout<<"circles.size()="<<circles.size()<<endl;
    for( size_t i =0; i< circles.size(); i++)
    {
        Vec3i c = circles[i];
        circle(colorImg, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 5, 8);

    }



    namedWindow("Circles detected", WINDOW_AUTOSIZE );
    imshow("Circles detected", greyImg);
    namedWindow("Original image", WINDOW_AUTOSIZE );
    imshow("Original image", colorImg);



    waitKey(0);
    return 0;
}
