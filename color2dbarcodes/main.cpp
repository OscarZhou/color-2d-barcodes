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
    IplImage* srcimg, *dstimg, *colordstimg;
    srcimg = cvLoadImage(argv[1], 0);
    dstimg = cvCreateImage(cvGetSize(srcimg), 8, 1);
    colordstimg = cvCreateImage(cvGetSize(srcimg), 8, 3);
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
    int i;

    //threshold(imgOri, imgPcd, 50, 255, THRESH_BINARY);

    cvCanny(srcimg, dstimg, 100, 100, 3);
    cvCvtColor(dstimg, colordstimg, CV_GRAY2BGR );

    lines = cvHoughLines2(dstimg, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 400);
    cout<<"lines->total="<<lines->total<<endl;

    set<float> float_set;
    for(i = 0; i < lines->total; i++)
    {
        float* line = (float*)cvGetSeqElem(lines, i);
        float rho = line[0];
        float theta = line[1];
        /*
        CvPoint pt1, pt2;
        double a = cos(theta), b = sin(theta);
        if(fabs(a) < 0.001)
        {
            pt1.x = pt2.x = cvRound(rho);
            pt1.y = 0;
            pt2.y = colordstimg->height;
        }
        else if(fabs(b) < 0.001)
        {
            pt1.y = pt2.y = cvRound(rho);
            pt1.x = 0;
            pt2.x = colordstimg->width;
        }
        else
        {
            pt1.x = 0;
            pt1.y = cvRound(rho/b);
            pt2.x = cvRound(rho/a);
            pt2.y = 0;
        }


        cvLine(colordstimg, pt1, pt2, CV_RGB(255, 0, 0), 2, 8);
        */
        float_set.insert(theta);

    }
    cout<<"float_set.size()="<<float_set.size()<<endl;


    float theta1 = 0.0;
    for(std::set<float>::iterator it=float_set.begin(); it!=float_set.end(); it++)
    {
        cout<<"theta="<<*it<<endl;
        theta1 = *it;
    }


    /* rotate the circle according to the theta */
    Mat rotmatrix = getRotationMatrix2D(Point(circles[0][0], circles[0][1]), theta1*180/CV_PI, 1);
    Mat affineImg;
    warpAffine(colorImg, affineImg, rotmatrix, colorImg.size());
    /*
    namedWindow("Before Rotation", WINDOW_AUTOSIZE );
    imshow("Before Rotation", colorImg);
    namedWindow("After Rotation", WINDOW_AUTOSIZE );
    imshow("After Rotation", affineImg);
    */


    //cvNamedWindow("Source", WINDOW_AUTOSIZE   );
    //cvShowImage("Source", srcimg);
    cvNamedWindow("Hough", WINDOW_AUTOSIZE  );
    cvShowImage("Hough", dstimg);

    //namedWindow("Circles detected", WINDOW_AUTOSIZE );
    //imshow("Circles detected", greyImg);
    namedWindow("Original image", WINDOW_AUTOSIZE );
    imshow("Original image", colorImg);



    waitKey(0);
    return 0;
}
