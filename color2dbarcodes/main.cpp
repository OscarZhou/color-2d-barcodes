#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "decoding.h"
#include <set>
#include <utility>

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

    /*  get a center of a circle */
    Mat colorImg = imread(argv[1], 1);
    if( colorImg.empty() )
    {
        printf("cannot open the image \n");
        exit(0);
    }

    Mat greyImg;
    cvtColor(colorImg, greyImg, CV_RGB2GRAY);

    medianBlur(greyImg, greyImg, 5);
    GaussianBlur(greyImg, greyImg, cv::Size(7, 7), 3, 3); // this function is very important
    vector<Vec3f> circles;
    cout<<"greyImg.rows/4="<<greyImg.rows/5<<endl;
    HoughCircles(greyImg, circles, CV_HOUGH_GRADIENT, 1, greyImg.rows/5, 200, 100, 200, 500);
    cout<<"circles.size()="<<circles.size()<<endl;
    for( size_t i =0; i< circles.size(); i++)
    {
        Vec3i c = circles[i];
        circle(colorImg, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 5, 8);
        /* paint the certer of the circle in order to observe the result of detection more obviously */
        int j = 0;
        for(j = 0; j< 5; j++)
        {
            MpixelR(colorImg, c[0], c[1]-j) = 0;
            MpixelG(colorImg, c[0], c[1]-j) = 0;
            MpixelB(colorImg, c[0], c[1]-j) = 0;

            MpixelR(colorImg, c[0]-j, c[1]) = 0;
            MpixelG(colorImg, c[0]-j, c[1]) = 0;
            MpixelB(colorImg, c[0]-j, c[1]) = 0;

            MpixelR(colorImg, c[0], c[1]+j) = 0;
            MpixelG(colorImg, c[0], c[1]+j) = 0;
            MpixelB(colorImg, c[0], c[1]+j) = 0;

            MpixelR(colorImg, c[0]+j, c[1]) = 0;
            MpixelG(colorImg, c[0]+j, c[1]) = 0;
            MpixelB(colorImg, c[0]+j, c[1]) = 0;
        }
    }

    /* find a line */
    Mat srcimgforline, dstimageforline, colorimgforline;
    srcimgforline = colorImg.clone();

    Canny( srcimgforline, dstimageforline, 50, 200, 3 );
    cvtColor( dstimageforline, colorimgforline, CV_GRAY2BGR );

    vector<Vec4i> lines1;
    HoughLinesP( dstimageforline, lines1, 1, CV_PI/180, 100, 200, 10 );

    //vector<float> angles;
    cout<<"lines.size()="<<lines1.size()<<endl;
    float angle;
    for( size_t i = 0; i < lines1.size(); i++ )
    {
        float x = lines1[i][2] - lines1[i][0];
        float y = lines1[i][3] - lines1[i][1];

        angle = round(atan2(y, x) * 180 / CV_PI);
        cout<<"|||||||||"<<angle<<endl;

        line( colorimgforline, Point(lines1[i][0], lines1[i][1]),
            Point(lines1[i][2], lines1[i][3]), Scalar(0,0,255), 1, 8 );


        //cout<<sqrt(y*y + x * x)<<endl;
        //pcount= std::make_pair()

        //angles.push_back(angle);

    }
    namedWindow("Original image1", WINDOW_AUTOSIZE );
    imshow("Original image1", colorimgforline);


    /* rotate the circle according to the theta */
    Mat rotmatrix = getRotationMatrix2D(Point(circles[0][0], circles[0][1]), angle, 1);
    Mat affineImg;
    warpAffine(colorImg, affineImg, rotmatrix, colorImg.size());

    namedWindow("Before Rotation", WINDOW_AUTOSIZE );
    imshow("Before Rotation", colorImg);
    namedWindow("After Rotation", WINDOW_AUTOSIZE );
    imshow("After Rotation", affineImg);



    //cvNamedWindow("Source", WINDOW_AUTOSIZE   );
    //cvShowImage("Source", srcimg);
    //cvNamedWindow("Hough", WINDOW_AUTOSIZE  );
    //cvShowImage("Hough", dstimg);

    //namedWindow("Circles detected", WINDOW_AUTOSIZE );
    //imshow("Circles detected", greyImg);
    //namedWindow("Original image", WINDOW_AUTOSIZE );
    //imshow("Original image", colorImg);



    waitKey(0);
    return 0;
}
