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

struct s_color
{
    bool b;
    bool g;
    bool r;

};

/*                       black      red       green      blue       white        cyan         magenta     yellow*/
//static s_block colors[8]={{0,0,0}, {180,0,0}, {0,180,0},{0,0,180},{180,180,180},{0,180,180},{180,0,180},{180,180,0}};

static s_color b_colors[24] = {{false, false, false},
                        {true, true, true},
                        {false, false, true},
                        {true, true, true},
                        {false, true, false},
                        {true, true, true},
                        {false, true, true},
                        {true, true, true},
                        {true, false, false},
                        {true, true, true},
                        {true, false, true},
                        {true, true, true},
                        {true, true, false},
                        {true, true, true},
                        {false, false, false},
                        {true, true, true},
                        {false, false, true},
                        {true, true, true},
                        {false, true, false},
                        {true, true, true},
                        {false, true, true},
                        {true, true, true},
                        {true, false, false},
                        {true, true, true}};

static int numofblockinrow[24] = {4, 8, 10, 12, 13, 15, 16, 17, 18, 18, 19, 20, 20, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 23};

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



    /* align the center of the circle */
    int icol = 0;//center.x-30;
    //int irow = 0;//center.y-30;
    //int k;
    //for(k=center.x-30; k<center.x+30; k++)
    //{
    //    s_block
    //}

    /*find the blocks contains the valid information */

    /*
    int irow;
    int offset[23] = {0}; // the value 20 is not fixed

    for(irow=0; irow<23; irow++)
    {
        offset[irow] = 15;
    }

    Point tmpCenter = Point(center.x, center.y-offset[22]);  // initial value is the neighbor of the center
    for(irow=22; irow>=0; irow--)
    {
        s_block tmpPt[3];
        bool bflag = true;
        int j=0;
        //tmpPt[0] stands for middle line, tmpPt[1] stands for top line, tmpPt[2] stands for bottom line
        int tmpoffset[3] = {0, -1, 1};
        for(j=0; j<3; j++)
        {
            tmpPt[j].r = MpixelR(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]);
            tmpPt[j].g = MpixelG(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]);
            tmpPt[j].b = MpixelB(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]);
            bool br = (tmpPt[j].r >= 180);
            bool bg = (tmpPt[j].g >= 180);
            bool bb = (tmpPt[j].b >= 180);


            if((br == b_colors[irow].r && bg == b_colors[irow].g && bb == b_colors[irow].b))
            {
                offset[irow] = offset[irow] + 15;
                break;
            }
            if((br == b_colors[irow-1].r && bg == b_colors[irow-1].g && bb == b_colors[irow-1].b))
            {

                if(j == 1 && !bflag)
                {
                    offset[irow-1] = offset[irow] - 15;
                    break;
                }
                else if(j == 2 && !bflag)
                {
                    offset[irow-1] = offset[irow] + 15;
                    break;
                }
                else
                {
                    offset[irow-1] = offset[irow-1] + offset[irow];
                    bflag = false;
                }
            }
        }

        tmpCenter = Point(center.x, center.y-offset[irow-1]);

        MpixelR(affineImg, tmpCenter.x, tmpCenter.y) = 0;
        MpixelG(affineImg, tmpCenter.x, tmpCenter.y) = 0;
        MpixelB(affineImg, tmpCenter.x, tmpCenter.y) = 0;


    }

    int loop=0;
    for(loop=0; loop<24; loop++)
    {
        cout<<"row["<<loop<<"]="<<offset[loop]<<endl;
    }


    for(irow=0; irow<23; irow++)
    {

    }
    */

    /* decode  emulator 2 square */
    Point pt1 = Point(center.x-20, center.y);//magenta
    Point pt2 = Point(center.x+10, center.y);//red
    s_block b1, b2;
    b1.r = MpixelR(affineImg, pt1.x, pt1.y)>180 ?1:0;
    b1.g = MpixelG(affineImg, pt1.x, pt1.y)>180 ?1:0;
    b1.b = MpixelB(affineImg, pt1.x, pt1.y)>180 ?1:0;

    b2.r = MpixelR(affineImg, pt2.x, pt2.y)>180 ?1:0;
    b2.g = MpixelG(affineImg, pt2.x, pt2.y)>180 ?1:0;
    b2.b = MpixelB(affineImg, pt2.x, pt2.y)>180 ?1:0;
    cout<<b1.r<<", "<<b1.g<<", "<<b1.b<<endl;
    int mask = 1<<5 | 1<<4 | 1<<3 | 1<<2 | 1<<1 | 1<<0;
    int bb1 = (b1.r<<5 | b1.g << 4| b1.b<<3) & mask;
    int bb2 = (b2.r<<2 | b2.g << 1| b2.b<<0) & mask;
    cout<<bb1<<endl;
    char x = encodingarray[bb1 | bb2];
    cout<<"x= "<<x<<endl;

    MpixelR(affineImg, pt1.x, pt1.y) = 0;
    MpixelG(affineImg, pt1.x, pt1.y) = 0;
    MpixelB(affineImg, pt1.x, pt1.y) = 0;


    MpixelR(affineImg, pt2.x, pt2.y) = 0;
    MpixelG(affineImg, pt2.x, pt2.y) = 0;
    MpixelB(affineImg, pt2.x, pt2.y) = 0;

    //namedWindow("Before Rotation", WINDOW_NORMAL );
    //imshow("Before Rotation", colorImg);
    namedWindow("After Rotation", WINDOW_AUTOSIZE);
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



