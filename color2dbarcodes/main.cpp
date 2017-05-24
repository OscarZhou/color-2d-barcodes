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

struct s_block
{
    int b;
    int g;
    int r;
};


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

    medianBlur(greyImg, greyImg, 7);

    //namedWindow("Median", WINDOW_AUTOSIZE  );
    //imshow("Median", greyImg);

    GaussianBlur(greyImg, greyImg, cv::Size(3, 3), 3, 3); // this function is very important
    //namedWindow("GaussianBlur", WINDOW_AUTOSIZE  );
    //imshow("GaussianBlur", greyImg);

    vector<Vec3f> circles;
    cout<<"greyImg.rows/8="<<greyImg.rows/5<<endl;
    HoughCircles(greyImg, circles, CV_HOUGH_GRADIENT, 1, greyImg.rows/8, 200, 100, 350, 0);
    cout<<"circles.size()="<<circles.size()<<endl;
    if(circles.size() < 1)
    {
        cout<<"can't recognize the circle !!!!!"<<endl;
        exit(0);
    }
    else if(circles.size() > 1)
    {
        vector<int > ptx, pty;
        for( size_t i =0; i< circles.size(); i++)
        {
            Vec3i c = circles[i];
            ptx.push_back(c[0]);
            pty.push_back(c[1]);

        }

        //int sumx = accumulate(ptx.begin(), ptx.end(), 0);
        //int sumy = accumulate(pty.begin(), pty.end(), 0);
        //circle(colorImg, Point(sumx/ptx.size(), sumy/pty.size()), c[2], Scalar(0, 0, 255), 5, 8);
        /* paint the certer of the circle in order to observe the result of detection more obviously */
        /*
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
        */
    }

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

    cout<<"lines.size()="<<lines1.size()<<endl;
    float angle;
    for( size_t i = 0; i < lines1.size(); i++ )
    {
        float x = lines1[i][2] - lines1[i][0];
        float y = lines1[i][3] - lines1[i][1];
        angle = round(atan2(y, x) * 180 / CV_PI);

        //line( colorimgforline, Point(lines1[i][0], lines1[i][1]),
        //    Point(lines1[i][2], lines1[i][3]), Scalar(0,0,255), 1, 8 );

    }
    /* rotate the circle according to the theta */
    Point center;
    center.x = circles[0][0];
    center.y = circles[0][1];
    cout<<"center = ("<<center.x<<", "<<center.y<<")"<<endl;
    Mat rotmatrix = getRotationMatrix2D(Point(center.x, center.y), angle, 1);
    Mat affineImg;
    warpAffine(colorImg, affineImg, rotmatrix, colorImg.size());



    /* detemine whether the circle is upright */
    s_block top, right, bottom, left;
    top.r = MpixelR(affineImg, center.x, center.y-20);
    top.g = MpixelG(affineImg, center.x, center.y-20);
    top.b = MpixelB(affineImg, center.x, center.y-20);
    cout<<"top is "<<top.r<<", "<< top.g <<", "<<top.b<<endl;


    right.r = MpixelR(affineImg, center.x+20, center.y);
    right.g = MpixelG(affineImg, center.x+20, center.y);
    right.b = MpixelB(affineImg, center.x+20, center.y);
    cout<<"right is "<<right.r<<", "<< right.g <<", "<<right.b<<endl;

    bottom.r = MpixelR(affineImg, center.x, center.y+20);
    bottom.g = MpixelG(affineImg, center.x, center.y+20);
    bottom.b = MpixelB(affineImg, center.x, center.y+20);
    cout<<"bottom is "<<bottom.r<<", "<< bottom.g <<", "<<bottom.b<<endl;

    left.r = MpixelR(affineImg, center.x-20, center.y);
    left.g = MpixelG(affineImg, center.x-20, center.y);
    left.b = MpixelB(affineImg, center.x-20, center.y);
    cout<<"left is "<<left.r<<", "<< left.g <<", "<<left.b<<endl;

    bool bTop = (top.r > 128 && top.g < 128 && top.b < 128);
    bool bRight = (right.r > 128 && right.g < 128 && right.b > 128);
    bool bBottom = (bottom.r > 128 && bottom.g < 128 && bottom.b > 128);
    bool bLeft = (left.r > 128 && left.g < 128 && left.b < 128);
    cout<<bTop<<",|| "<<bRight<<", "<<bBottom<<", "<<bLeft<<endl;
    if( bTop && !bRight && bBottom && !bLeft)
    {
        angle = 270;
    }
    else if( !bTop && !bRight && !bBottom && !bLeft )
    {
        angle = 180;
    }
    else if( !bTop && bRight && !bBottom && bLeft )
    {
        angle = 90;
    }
    else
    {
        angle = 0;
    }

    cout<<"the angle of the rotation is "<<angle<<endl;
    rotmatrix = getRotationMatrix2D(Point(center.x, center.y), angle, 1);
    warpAffine(affineImg, affineImg, rotmatrix, colorImg.size());

    /**/


    //namedWindow("Before Rotation", WINDOW_NORMAL );
    //imshow("Before Rotation", colorImg);
    namedWindow("After Rotation", WINDOW_AUTOSIZE  );
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
