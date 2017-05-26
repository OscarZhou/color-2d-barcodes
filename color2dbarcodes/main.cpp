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


#define MASK (1<<5 | 1<<4 | 1<<3 | 1<<2 | 1<<1 | 1<<0)


struct s_block
{
    int b;
    int g;
    int r;
};

struct s_color
{
    bool r;
    bool g;
    bool b;
};

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

static int numofblockinrow[23] = {4, 8, 10, 12, 13,
                                15, 16, 17, 18, 18,
                                19, 20, 20, 21, 21,
                                22, 22, 22, 22, 23,
                                23, 23, 23};

int getCircle(Mat colorImg, Vec3i& ccl);
int getAngle(Mat colorImg);
void rotateCircle(Mat colorImg, Point center, int angle, Mat& affineImg);
char decode(Mat affineImg, Point pt1, Point pt2);


void printpoint(Mat colorImg, Point pt)
{

    MpixelR(colorImg, pt.x, pt.y) = 255;
    MpixelG(colorImg, pt.x, pt.y) = 0;
    MpixelB(colorImg, pt.x, pt.y) = 0;

    //cout<<"pt=("<<pt.x<<", "<<pt.y<<")"<<endl;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("needs an image\n");
        exit(0);
    }

    /*  read an image */
    Mat colorImg = imread(argv[1], 1);
    if( colorImg.empty() )
    {
        printf(" fail to read image \n");
        exit(0);
    }

    /*  find a circle */
    Vec3i biggest_circle;
    int ret = getCircle(colorImg, biggest_circle);
    if(ret < 0)
    {
        cout<<"can't recognize the circle !!!!!"<<endl;
        exit(0);
    }

    Point center;
    center.x = biggest_circle[0];
    center.y = biggest_circle[1];
    int radius = biggest_circle[2];

    /* get the angle */
    int angle= getAngle(colorImg);

    /* rotate the circle according to the theta */
    Mat affineImg;
    rotateCircle(colorImg, center, angle, affineImg);

    namedWindow("1rotat", WINDOW_AUTOSIZE);
    imshow("1rotat", affineImg);

    /* detemine whether the circle is upright */
    s_color col_top, col_right, col_bottom, col_left;
    col_top.r = MpixelR(affineImg, center.x, center.y-20)>180 ?true:false;
    col_top.g = MpixelG(affineImg, center.x, center.y-20)>180 ?true:false;
    col_top.b = MpixelB(affineImg, center.x, center.y-20)>180 ?true:false;
    //cout<<"top is "<<col_top.r<<", "<< col_top.g <<", "<<col_top.b<<endl;

    col_right.r = MpixelR(affineImg, center.x+20, center.y)>180 ?true:false;
    col_right.g = MpixelG(affineImg, center.x+20, center.y)>180 ?true:false;
    col_right.b = MpixelB(affineImg, center.x+20, center.y)>180 ?true:false;
    //cout<<"right is "<<col_right.r<<", "<< col_right.g <<", "<<col_right.b<<endl;

    col_bottom.r = MpixelR(affineImg, center.x, center.y+20)>180 ?true:false;
    col_bottom.g = MpixelG(affineImg, center.x, center.y+20)>180 ?true:false;
    col_bottom.b = MpixelB(affineImg, center.x, center.y+20)>180 ?true:false;
    //cout<<"bottom is "<<col_bottom.r<<", "<< col_bottom.g <<", "<<col_bottom.b<<endl;

    col_left.r = MpixelR(affineImg, center.x-20, center.y)>180 ?true:false;
    col_left.g = MpixelG(affineImg, center.x-20, center.y)>180 ?true:false;
    col_left.b = MpixelB(affineImg, center.x-20, center.y)>180 ?true:false;
    //cout<<"left is "<<col_left.r<<", "<< col_left.g <<", "<<col_left.b<<endl;


    bool bTop = (col_top.r && !col_top.g && !col_top.b);
    bool bRight = (col_right.r && !col_right.g && col_right.b);
    bool bBottom = (col_bottom.r && !col_bottom.g && col_bottom.b);
    bool bLeft = (col_left.r && !col_left.g && !col_left.b);

    cout<<"------------->"<<bTop<<", "<<bRight<<", "<<bBottom<<", "<<bLeft<<endl;
    int symbol = 1;
    if(angle < 0) symbol = -1;
    if( bTop && !bRight && bBottom && !bLeft)
    {
        angle = 270 * symbol;
    }
    else if( !bTop && !bRight && !bBottom && !bLeft )
    {
        angle = 180 * symbol;
    }
    else if( !bTop && bRight && !bBottom && bLeft )
    {
        angle = 90 * symbol;
    }
    else
    {
        angle = 0 * symbol;
    }


    cout<<"the second angle of the rotation is "<<angle<<endl;
    Mat rotmatrix = getRotationMatrix2D(Point(center.x, center.y), angle, 1);
    warpAffine(affineImg, affineImg, rotmatrix, colorImg.size());

    //cout<<"center=("<<center.x<<", "<<center.y<<")"<<endl;
    namedWindow("2rotat", WINDOW_AUTOSIZE);
    imshow("2rotat", affineImg);


    /* align the center of the circle */
    // according to the radius to determine the times of for loop
    int min_col = center.x-round(radius/23), max_col = center.x+round(radius/23);//center.x-30;
    int min_row = center.y-round(radius/23), max_row = center.y+round(radius/23);//center.y-30;

    int k, bleft=0, btop=0;
    int length_row = 0;
    int length_col = 0;
    for(k=min_col; k<max_col; k++)
    {
        s_color color;
        color.r = MpixelR(affineImg, k, center.y)>180 ?true:false;
        color.g = MpixelG(affineImg, k, center.y)>180 ?true:false;
        color.b = MpixelB(affineImg, k, center.y)>180 ?true:false;

        if((color.r && color.g && color.b) )
        {
            if(length_col == 0)
            {
                bleft = k;
            }
            length_col++;
        }
        else continue;

    }

    for(k=min_row; k<max_row; k++)
    {
        s_color color;
        color.r = MpixelR(affineImg, center.x, k)>180 ?true:false;
        color.g = MpixelG(affineImg, center.x, k)>180 ?true:false;
        color.b = MpixelB(affineImg, center.x, k)>180 ?true:false;

        if((color.r && color.g && color.b))
        {
            if(length_row == 0)
            {
                btop = k;
            }
            length_row++;
        }
        else continue;

    }
    //cout<<"boundary position=("<<bleft<<", "<<btop<<")"<<endl;
    Point pt3 = Point(bleft, center.y);
    Point pt4 = Point(center.x, btop);
    printpoint(affineImg, pt3);
    printpoint(affineImg, pt4);


    center.x = bleft + round(length_col / 2.0) -1;
    center.y = btop + round(length_row / 2.0) -1;
    //cout<<"center=("<<center.x<<", "<<center.y<<")"<<endl;
    printpoint(affineImg, center);



    /* find a pattern */
    int irow;
    int offset[23] = {0}; // the value 20 is not fixed

    for(irow=0; irow<23; irow++)
    {
        offset[irow] = length_row+2;
    }
    cout<<"offset="<<length_row<<endl;
    Point tmpCenter;  // initial value is the neighbor of the center
    //printpoint(affineImg, tmpCenter);
    for(irow=0; irow<23; irow++)
    {
        //cout<<"~~~~@@@ irow="<<irow<<endl;
        if(irow != 0)
        {
            offset[irow] =  offset[irow] + offset[irow-1];
        }
        tmpCenter = Point(center.x, center.y-offset[irow]);
        //s_block tmpPt[3];
        bool bflag = true;
        int j=0;
        //tmpPt[0] stands for middle line, tmpPt[1] stands for top line, tmpPt[2] stands for bottom line
        int tmpoffset[5] = {0, -2, 2, -1, 1};
        for(j=0; j<5; j++)
        {
            //tmpPt[j].r = MpixelR(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]);
            //tmpPt[j].g = MpixelG(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]);
            //tmpPt[j].b = MpixelB(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]);
            //bool br = (tmpPt[j].r >= 180);
            //bool bg = (tmpPt[j].g >= 180);
            //bool bb = (tmpPt[j].b >= 180);


            bool br = MpixelR(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]) >= 180 ? true: false;
            bool bg = MpixelG(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]) >= 180 ? true: false;
            bool bb = MpixelB(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]) >= 180 ? true: false;


            //if((br == b_colors[irow].r && bg == b_colors[irow].g && bb == b_colors[irow].b))
            //{
            //    offset[irow] = offset[irow] + length_row/2;
            //    break;
            //}

            if(!(br == b_colors[22-irow].r && bg == b_colors[22-irow].g && bb == b_colors[22-irow].b))
            {
                //cout<<"irow="<<irow<<", j="<<j<<endl;

                if(j == 3 && !bflag)
                {
                    offset[irow] = offset[irow] - length_row/2;
                    break;
                }
                else if(j == 4 && !bflag)
                {
                    offset[irow] = offset[irow] + length_row/2;
                    break;
                }
                else if(j == 1)
                {
                    offset[irow] = offset[irow] - length_row*3/4;
                    //cout<<"!!!"<<endl;
                    break;
                }
                else if(j == 2 )
                {
                    offset[irow] = offset[irow] + length_row*3/4;
                    //cout<<"!!!"<<endl;
                    break;
                }

                else
                {
                    bflag = false;
                }
            }
        }

        //tmpCenter = Point(center.x, center.y-offset[irow]);
        //printpoint(affineImg, Point(center.x, center.y-offset[irow]));

    }
    /*
    int loop=0;
    for(loop=0; loop<24; loop++)
    {
        cout<<"row["<<loop<<"]="<<offset[loop]<<endl;
    }
    */



    /* find the blocks contains the valid information  */
    vector<char> text;
    for(irow=22; irow>=0; irow--)
    {
        //cout<<"~~irow="<<irow<<endl;
        int icol;
        Point pt1, pt2;
        //cout<<"~~numofblockinrow[22-irow]="<<numofblockinrow[22-irow]<<endl;
        //break;
        int counter = numofblockinrow[22-irow] * 2;
        icol = numofblockinrow[22-irow]-1;
        bool left_flag = true;
        do
        {
            if(icol==0)
            {
                if(numofblockinrow[22-irow]%2==1)
                {
                    pt1 = Point(center.x-offset[icol], center.y-offset[irow]);
                    pt2 = Point(center.x+offset[icol], center.y-offset[irow]);
                    text.push_back(decode(affineImg, pt1, pt2));
                    //printpoint(affineImg, pt1);
                    //printpoint(affineImg, pt2);
                    counter -= 2;
                    icol += 2;
                }
                else
                {
                    icol = 1;
                    //left_flag = false;
                    //cout<<"1 icol="<<icol<<endl;
                    //continue;
                }
                left_flag = false;
                //icol += 2;
                //cout<<"2 icol="<<icol<<endl;
                continue;
            }
            if(left_flag)
            {
                pt1 = Point(center.x-offset[icol], center.y-offset[irow]);
                pt2 = Point(center.x-offset[icol-1], center.y-offset[irow]);
                text.push_back(decode(affineImg, pt1, pt2));
                //printpoint(affineImg, pt1);
                //printpoint(affineImg, pt2);
                counter -= 2;
                //cout<<"3 icol="<<icol<<endl;
                if((numofblockinrow[22-irow]%2==0) && icol ==1)
                {
                    icol = 0;
                    continue;
                }
                icol -= 2;

            }
            else
            {
                pt1 = Point(center.x+offset[icol-1], center.y-offset[irow]);
                pt2 = Point(center.x+offset[icol], center.y-offset[irow]);
                text.push_back(decode(affineImg, pt1, pt2));
                //printpoint(affineImg, pt1);
                //printpoint(affineImg, pt2);
                icol += 2;
                counter -= 2;
                //cout<<"4 icol="<<icol<<endl;
            }

        }while(counter);
    }
    //
    for(irow=0; irow<23; irow++)
    {
        //cout<<"~~irow="<<irow<<endl;
        int icol;
        Point pt1, pt2;
        //cout<<"~~numofblockinrow[irow]="<<numofblockinrow[22-irow]<<endl;
        //break;
        int counter = numofblockinrow[22-irow] * 2;
        icol = numofblockinrow[22-irow]-1;
        bool left_flag = true;
        do
        {
            if(icol==0)
            {
                if(numofblockinrow[22-irow]%2==1)
                {
                    pt1 = Point(center.x-offset[icol], center.y+offset[irow]);
                    pt2 = Point(center.x+offset[icol], center.y+offset[irow]);
                    text.push_back(decode(affineImg, pt1, pt2));
                    //printpoint(affineImg, pt1);
                    //printpoint(affineImg, pt2);
                    counter -= 2;
                    icol += 2;
                }
                else
                {
                    icol = 1;
                    //left_flag = false;
                    //cout<<"1 icol="<<icol<<endl;
                    //continue;
                }
                left_flag = false;
                //icol += 2;
                //cout<<"2 icol="<<icol<<endl;
                continue;
            }
            if(left_flag)
            {
                pt1 = Point(center.x-offset[icol], center.y+offset[irow]);
                pt2 = Point(center.x-offset[icol-1], center.y+offset[irow]);
                text.push_back(decode(affineImg, pt1, pt2));
                //printpoint(affineImg, pt1);
                //printpoint(affineImg, pt2);
                counter -= 2;
                //cout<<"3 icol="<<icol<<endl;
                if((numofblockinrow[22-irow]%2==0) && icol ==1)
                {
                    icol = 0;
                    continue;
                }
                icol -= 2;

            }
            else
            {
                pt1 = Point(center.x+offset[icol-1], center.y+offset[irow]);
                pt2 = Point(center.x+offset[icol], center.y+offset[irow]);
                text.push_back(decode(affineImg, pt1, pt2));
                //printpoint(affineImg, pt1);
                //printpoint(affineImg, pt2);
                icol += 2;
                counter -= 2;
                //cout<<"4 icol="<<icol<<endl;
            }

        }while(counter);
    }
    for(std::vector<char>::iterator it=text.begin(); it!=text.end(); it++)
    {
        cout<<*it;
    }


    cout<<"r="<<radius/23<<endl;
    //namedWindow("Before Rotation", WINDOW_NORMAL );
    //imshow("Before Rotation", colorImg);
    //namedWindow("After Rotation", WINDOW_AUTOSIZE);
    //imshow("After Rotation", affineImg);

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

/************************************************************************
*
* Function Description: get a circle
* Parameter Description:
* * coloImg : input image
* * ccl: output biggest circle
*
*************************************************************************/
int getCircle(Mat colorImg, Vec3i& ccl)
{
    Mat greyImg;
    cvtColor(colorImg, greyImg, CV_RGB2GRAY);

    medianBlur(greyImg, greyImg, 7);
    GaussianBlur(greyImg, greyImg, cv::Size(3, 3), 3, 3);

    vector<Vec3f> circles;
    HoughCircles(greyImg, circles, CV_HOUGH_GRADIENT, 1, greyImg.rows/8, 200, 100, greyImg.rows/8, 0);
    if(circles.size() < 1)
    {
        return -1;
    }
    int circle_index = 0;
    int radius = 0;
    for( size_t i =0; i< circles.size(); i++)
    {
        Vec3i c = circles[i];
        if(i == 0 || c[2]>radius)
        {
            circle_index = i;
            radius = c[2];

        }
        //circle(colorImg, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 5, 8);
    }
    ccl = circles[circle_index];
    return 1;
}


/************************************************************************
*
* Function Description: get the aligned angle
* Parameter Description:
* * coloImg : input image
* * return value: angle
*
*************************************************************************/
int getAngle(Mat colorImg)
{
    float angle = 0;
    Mat srcimgforline, dstimageforline, colorimgforline;
    srcimgforline = colorImg.clone();

    Canny( srcimgforline, dstimageforline, 50, 200, 3 );
    cvtColor( dstimageforline, colorimgforline, CV_GRAY2BGR );

    vector<Vec4i> lines;
    HoughLinesP( dstimageforline, lines, 1, CV_PI/180, 100, 200, 100 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float x = lines[i][2] - lines[i][0];
        float y = lines[i][3] - lines[i][1];
        angle = round(atan2(y, x) * 180 / CV_PI);
        //line( colorimgforline, Point(lines1[i][0], lines1[i][1]),
        //    Point(lines1[i][2], lines1[i][3]), Scalar(0,0,255), 1, 8 );
        break;
    }
    return angle;
}

/************************************************************************
*
* Function Description: get the aligned angle
* Parameter Description:
* * coloImg : input image
* * return value: angle
*
*************************************************************************/
void rotateCircle(Mat colorImg, Point center, int angle, Mat& affineImg)
{
    Mat rotmatrix = getRotationMatrix2D(Point(center.x, center.y), angle, 1);
    warpAffine(colorImg, affineImg, rotmatrix, colorImg.size());
}

char decode(Mat affineImg, Point pt1, Point pt2)
{
    /* decode  emulator 2 square */
    //Point pt1 = Point(center.x-20, center.y);//magenta
    //Point pt2 = Point(center.x+10, center.y);//red
    s_block b1, b2;
    b1.r = MpixelR(affineImg, pt1.x, pt1.y)>180 ?1:0;
    b1.g = MpixelG(affineImg, pt1.x, pt1.y)>180 ?1:0;
    b1.b = MpixelB(affineImg, pt1.x, pt1.y)>180 ?1:0;

    b2.r = MpixelR(affineImg, pt2.x, pt2.y)>180 ?1:0;
    b2.g = MpixelG(affineImg, pt2.x, pt2.y)>180 ?1:0;
    b2.b = MpixelB(affineImg, pt2.x, pt2.y)>180 ?1:0;

    int bb1 = (b1.r<<5 | b1.g << 4| b1.b<<3) & MASK;
    int bb2 = (b2.r<<2 | b2.g << 1| b2.b<<0) & MASK;

    char x = encodingarray[bb1 | bb2];

    printpoint(affineImg, pt1);
    printpoint(affineImg, pt2);

    return encodingarray[bb1 | bb2];
}

