#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "decoding.h"
#include <set>
#include <utility>
#include <chrono>
#include <ctime>

using namespace std;
using namespace cv;
using namespace chrono;

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

struct s_blockinfo
{
    int width;
    int height;
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
void rotateCircle(Mat colorImg, Vec3i ccl, int angle, Mat& affineImg);
int getUprightDownAngle(Mat affineImg, Vec3i ccl);
void relocateCenterofCircle(Mat affineImg, Vec3i& ccl, s_blockinfo& block);
void findOffsetPattern(Mat affineImg, Vec3i ccl, int* offset, int length, s_blockinfo block);
vector<char> translate(Mat affineImg, Vec3i ccl, int* offest, int length);
char decode(Mat affineImg, Point pt1, Point pt2);


void printpoint(Mat colorImg, Point pt)
{

    MpixelR(colorImg, pt.x, pt.y) = 255;
    MpixelG(colorImg, pt.x, pt.y) = 0;
    MpixelB(colorImg, pt.x, pt.y) = 0;

    //cout<<"pt=("<<pt.x<<", "<<pt.y<<")"<<endl;
}


Mat frame;//, image;
int main(int argc, char** argv)
{

    VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
    {
        cout << "Failed to open camera" << endl;
        return 0;
    }
    cout << "Opened camera" << endl;
    namedWindow("WebCam", 1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    cap >> frame;
    printf("frame size %d %d \n",frame.rows, frame.cols);
    int key=0;

    double fps=0.0;
    while (1){
        system_clock::time_point start = system_clock::now();
        cap >> frame;

        if( frame.empty() )
            break;

        /*  find a circle */
        Vec3i biggest_circle;
        int ret = getCircle(frame, biggest_circle);
        if(ret < 0)
        {
            cout<<"can't recognize the circle !!!!!"<<endl;
            break;
        }

        /* get the angle */
        int angle= getAngle(frame);
        /* rotate the circle according to the theta */
        Mat affineImg;
        rotateCircle(frame, biggest_circle, angle, affineImg);

        /* detemine whether the circle is upright */
        angle = getUprightDownAngle(affineImg, biggest_circle);

        /* rotate circle in second time */
        rotateCircle(affineImg, biggest_circle, angle, affineImg);


        /* align the center of the circle */
        s_blockinfo blockinfo = {0 , 0};
        relocateCenterofCircle(affineImg, biggest_circle, blockinfo);
        printpoint(affineImg, Point(biggest_circle[0], biggest_circle[1]));

        /* find a pattern */
        int offset[23] = {0};
        findOffsetPattern(affineImg, biggest_circle, offset, 23, blockinfo);

        /* find the blocks contains the valid information  */
        vector<char> text = translate(affineImg, biggest_circle, offset, 23);
        /* print text*/
        for(std::vector<char>::iterator it=text.begin(); it!=text.end(); it++)
        {
            cout<<*it;
        }
        cout<<endl;





        char printit[100];
        sprintf(printit,"%2.1f",fps);
        putText(frame, printit, cvPoint(10,30), FONT_HERSHEY_PLAIN, 2, cvScalar(255,255,255), 2, 8);
        imshow("WebCam", frame);
        key=waitKey(1);
        if(key==113 || key==27) return 0;//either esc or 'q'

        system_clock::time_point end = system_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //fps = 1000000*10.0/seconds;
        fps = 1000000/seconds;
        cout << "frames " << fps << " seconds " << seconds << endl;
    }
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

    //medianBlur(greyImg, greyImg, 7);
    GaussianBlur(greyImg, greyImg, cv::Size(9, 9), 3, 3);

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
    /*
    for(int i=0; i<5 ; i++)
    {
        printpoint(colorImg, Point(ccl[0], ccl[1]-i));
        printpoint(colorImg, Point(ccl[0]+i, ccl[1]));
        printpoint(colorImg, Point(ccl[0], ccl[1]+i));
        printpoint(colorImg, Point(ccl[0]-i, ccl[1]));

    }
    */
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
        //line( colorimgforline, Point(lines[i][0], lines[i][1]),
        //    Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 1, 8 );
        break;
    }

    return angle;
}

/************************************************************************
*
* Function Description: rotate circle
* Parameter Description:
* * coloImg : input image
* * center : the center of the circle
* * angle : the aligned angle
* * affineImg : output image
*
*************************************************************************/
void rotateCircle(Mat colorImg, Vec3i ccl, int angle, Mat& affineImg)
{
    Point center = Point(ccl[0], ccl[1]);
    Mat rotmatrix = getRotationMatrix2D(Point(center.x, center.y), angle, 1);
    warpAffine(colorImg, affineImg, rotmatrix, colorImg.size());
}


/************************************************************************
*
* Function Description: get the angle which is upright down
* Parameter Description:
* * affineImg : input image
* * ccl : the biggest circle
* * return value : the angle which make circle upright
*
*************************************************************************/
int getUprightDownAngle(Mat affineImg, Vec3i ccl)
{
    int angle;
    Point center = Point(ccl[0], ccl[1]);
    int radius = ccl[2];
    int offset = radius/23;

    s_color col_top, col_right, col_bottom, col_left;
    col_top.r = MpixelR(affineImg, center.x, center.y-offset)>180 ?true:false;
    col_top.g = MpixelG(affineImg, center.x, center.y-offset)>180 ?true:false;
    col_top.b = MpixelB(affineImg, center.x, center.y-offset)>180 ?true:false;
    //printpoint(affineImg, Point(center.x, center.y-offset));
    //cout<<"top is "<<col_top.r<<", "<< col_top.g <<", "<<col_top.b<<endl;


    col_right.r = MpixelR(affineImg, center.x+offset, center.y)>180 ?true:false;
    col_right.g = MpixelG(affineImg, center.x+offset, center.y)>180 ?true:false;
    col_right.b = MpixelB(affineImg, center.x+offset, center.y)>180 ?true:false;
    //printpoint(affineImg, Point(center.x+offset, center.y));
    //cout<<"right is "<<col_right.r<<", "<< col_right.g <<", "<<col_right.b<<endl;

    col_bottom.r = MpixelR(affineImg, center.x, center.y+offset)>180 ?true:false;
    col_bottom.g = MpixelG(affineImg, center.x, center.y+offset)>180 ?true:false;
    col_bottom.b = MpixelB(affineImg, center.x, center.y+offset)>180 ?true:false;
    //printpoint(affineImg, Point(center.x, center.y+offset));
    //cout<<"bottom is "<<col_bottom.r<<", "<< col_bottom.g <<", "<<col_bottom.b<<endl;

    col_left.r = MpixelR(affineImg, center.x-offset, center.y)>180 ?true:false;
    col_left.g = MpixelG(affineImg, center.x-offset, center.y)>180 ?true:false;
    col_left.b = MpixelB(affineImg, center.x-offset, center.y)>180 ?true:false;
    //printpoint(affineImg, Point(center.x-offset, center.y));
    //cout<<"left is "<<col_left.r<<", "<< col_left.g <<", "<<col_left.b<<endl;

    //this is the correct pattern
    bool bTop = (col_top.r && !col_top.g && !col_top.b);
    bool bRight = (col_right.r && !col_right.g && col_right.b);
    bool bBottom = (col_bottom.r && !col_bottom.g && col_bottom.b);
    bool bLeft = (col_left.r && !col_left.g && !col_left.b);

    //cout<<"------------->"<<bTop<<", "<<bRight<<", "<<bBottom<<", "<<bLeft<<endl;

    if( bTop && !bRight && bBottom && !bLeft)
    {
        angle = 90;
    }
    else if( !bTop && !bRight && !bBottom && !bLeft )
    {
        angle = 180;
    }
    else if( !bTop && bRight && !bBottom && bLeft )
    {
        angle = 270;
    }
    else
    {
        angle = 0;
    }

    return angle;
}


/************************************************************************
*
* Function Description: relocate the center of the circle for finding a offset pattern more accurate
* Parameter Description:
* * affineImg : input image
* * ccl : the biggest circle
* * block : store the info of the block, like width and length
* * return value : the angle which make circle upright
*
*************************************************************************/
void relocateCenterofCircle(Mat affineImg, Vec3i& ccl, s_blockinfo& block)
{
    Point center = Point(ccl[0], ccl[1]);
    int radius = ccl[2];

    int min_col = center.x-round(radius/23), max_col = center.x+round(radius/23);//center.x-30;
    int min_row = center.y-round(radius/23), max_row = center.y+round(radius/23);//center.y-30;

    int k, bleft=0, btop=0;
    int length_row = 0;
    int length_col = 0;
    for(k=min_col; k<max_col; k++) //calculate the numbers of blocks in horizotal direction
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

    for(k=min_row; k<max_row; k++) //calculate the numbers of blocks in vertical direction
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

    ccl[0] = bleft + round(length_col / 2.0)-1;
    ccl[1] = btop + round(length_row / 2.0)-1;
    block.width = length_col;
    block.height = length_row;
    //printpoint(affineImg, center);
}

/************************************************************************
*
* Function Description: relocate the center of the circle for finding a offset pattern more accurate
* Parameter Description:
* * affineImg : input image
* * ccl : the biggest circle
* * offset : output value, offset pattern, very important
* * length : the size of the array offset
* * block : store the info of the block, like width and length
*
*************************************************************************/
void findOffsetPattern(Mat affineImg, Vec3i ccl, int* offset, int length, s_blockinfo block)
{
    Point center = Point(ccl[0], ccl[1]);
    int radius = ccl[2];

    int irow;
    for(irow=0; irow<length; irow++)
    {
        offset[irow] = block.height+2;

    }

    Point tmpCenter;
    int tmpoffset[5] = {0, -2, 2, -1, 1};
    for(irow=0; irow<length; irow++)
    {
        if(irow != 0)
        {
            offset[irow] =  offset[irow] + offset[irow-1];
        }
        tmpCenter = Point(center.x, center.y-offset[irow]);
        bool bflag = true;
        int j=0;
        for(j=0; j<5; j++)
        {
            bool br = MpixelR(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]) >= 180 ? true: false;
            bool bg = MpixelG(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]) >= 180 ? true: false;
            bool bb = MpixelB(affineImg, tmpCenter.x, tmpCenter.y + tmpoffset[j]) >= 180 ? true: false;

            if(!(br == b_colors[22-irow].r && bg == b_colors[22-irow].g && bb == b_colors[22-irow].b))
            {
                if(j == 3 && !bflag)
                {
                    offset[irow] = offset[irow] - block.height/2;
                    break;
                }
                else if(j == 4 && !bflag)
                {
                    offset[irow] = offset[irow] + block.height/2;
                    break;
                }
                else if(j == 1)
                {
                    offset[irow] = offset[irow] - block.height/2;
                    break;
                }
                else if(j == 2 )
                {
                    offset[irow] = offset[irow] + block.height/2;
                    break;
                }
                else
                {
                    bflag = false;
                }
            }

        }
        printpoint(affineImg, tmpCenter);
    }

}

/************************************************************************
*
* Function Description: translate the 2dbarcode
* Parameter Description:
* * affineImg : input image
* * ccl : the biggest circle
* * offset : output value, offset pattern, very important
* * length : the size of the array offset
* * return value : decoded text
*
*************************************************************************/
vector<char> translate(Mat affineImg, Vec3i ccl, int* offset, int length)
{
    vector<char> txt;

    Point center = Point(ccl[0], ccl[1]);
    int radius = ccl[2];

    int irow=0;
    for(irow=length-1; irow>=0; irow--)
    {

        Point pt1, pt2;
        int blockNum = numofblockinrow[(length-1)-irow];
        int counter = blockNum * 2;  // the number of iterating block
        bool stay_left_area = true; // the current area of iterating block, true:left area, false: right area
        int icol = blockNum-1;  // the start position for reading

        do
        {
            if(icol==0)
            {
                if(blockNum%2==1)
                {
                    pt1 = Point(center.x-offset[icol], center.y-offset[irow]);
                    pt2 = Point(center.x+offset[icol], center.y-offset[irow]);
                    txt.push_back(decode(affineImg, pt1, pt2));
                    counter -= 2;
                    icol += 2;
                }
                else
                {
                    icol = 1; //if the blockNum is even, read right area from index icol-1
                }
                stay_left_area = false;
                continue;
            }
            if(stay_left_area)
            {
                pt1 = Point(center.x-offset[icol], center.y-offset[irow]);
                pt2 = Point(center.x-offset[icol-1], center.y-offset[irow]);
                txt.push_back(decode(affineImg, pt1, pt2));
                counter -= 2;
                if((blockNum%2==0) && icol ==1)
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
                txt.push_back(decode(affineImg, pt1, pt2));
                icol += 2;
                counter -= 2;
            }

        }while(counter);
    }

    for(irow=0; irow<length; irow++)
    {
        Point pt1, pt2;
        int blockNum = numofblockinrow[(length-1)-irow];
        int counter = blockNum * 2;  // the number of iterating block
        bool stay_left_area = true; // the current area of iterating block, true:left area, false: right area
        int icol = blockNum-1;  // the start position for reading

        do
        {
            if(icol==0)
            {
                if(blockNum%2==1)
                {
                    pt1 = Point(center.x-offset[icol], center.y+offset[irow]);
                    pt2 = Point(center.x+offset[icol], center.y+offset[irow]);
                    txt.push_back(decode(affineImg, pt1, pt2));
                    counter -= 2;
                    icol += 2;
                }
                else
                {
                    icol = 1;
                }
                stay_left_area = false;
                continue;
            }
            if(stay_left_area)
            {
                pt1 = Point(center.x-offset[icol], center.y+offset[irow]);
                pt2 = Point(center.x-offset[icol-1], center.y+offset[irow]);
                txt.push_back(decode(affineImg, pt1, pt2));
                counter -= 2;
                if((blockNum%2==0) && icol ==1)
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
                txt.push_back(decode(affineImg, pt1, pt2));
                icol += 2;
                counter -= 2;
            }

        }while(counter);
    }
    return txt;
}

/************************************************************************
*
* Function Description: decoding function
* Parameter Description:
* * affineImg : input image
* * pt1 : left pixel
* * pt2 : right pixel
* * return : decoded one char
*
*************************************************************************/
char decode(Mat affineImg, Point pt1, Point pt2)
{
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

    //printpoint(affineImg, pt1);
    //printpoint(affineImg, pt2);

    return encodingarray[bb1 | bb2];
}
