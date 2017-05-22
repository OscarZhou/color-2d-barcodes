#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <set>
#include <ObjUnit.h>
#include <time.h>


using namespace std;
using namespace cv;


#define Mpixel(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)]

#define MpixelB(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())]
#define MpixelG(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())+1]
#define MpixelR(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())+2]


typedef vector<point_set> obj_vector;
//typedef vector<ObjUnit> obj_vector;


int get_num_of_object(obj_vector vectorOfObj);
int compare_for_median(const void * a, const void * b);
Mat do_median_filter(Mat imageOri);
int count_blob(Mat image);
Mat color_object(Mat imageOri, obj_vector vectorOfObj, int numOfObj);
Mat SmoothingFilter(Mat image, double* mask, int side);

int g_counter=0;

int main(int argc, char** argv)
{
    if(argc!=2) {
        cout<<"needs 2 argument, e.g.image.jpg"<<endl;
        exit(0);
	}

    namedWindow("Figure1", 0);
    namedWindow("Figure2", 0);

    /************************************************************************
    *
    * Read the original image and then create space for the two Mat being used
    *
    *************************************************************************/

	Mat imageOri = imread(argv[1], IMREAD_GRAYSCALE);


    clock_t  clockBegin, clockEnd;
    clockBegin = clock();

    Mat imageRet = do_median_filter(imageOri);
    //for(int i=0; i<10; i++)
      //  imageRet = do_median_filter(imageRet);

    clockEnd = clock();
    printf("do_median_filter takes %ld ms\n", clockEnd - clockBegin);




    cout<< imageOri.cols<<"   "<<imageOri.rows<<endl;
	imshow("Figure1", imageOri);
	imshow("Figure2", imageRet);

    /************************************************************************
    *
    * Binamise the filtered image
    *
    *************************************************************************/
    Mat imageThld;     // The original image which is obtained by read function
    threshold(imageRet, imageThld, 0 ,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    namedWindow("Figure3", 0);
	imshow("Figure3", imageThld);


    imageRet = do_median_filter(imageThld);

    clockBegin = clock();
    int num = count_blob(imageRet);


    clockEnd = clock();
    printf("count_blob takes %ld s\n", clockEnd - clockBegin);


    cout<<"The number of objects is "<<num<<endl;
    cout<<"The number of g_counter is "<<g_counter<<endl;

	waitKey(0);
    return 0;
}



/************************************************************************
*
* this function is used as qsort rule
*
*************************************************************************/
int compare_for_median(const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}

Mat do_median_filter(Mat imageOri)
{
    int pMask[9] = {0}; // Store the temperary data used for qsort
    Mat imageRet, imageExtention;
    imageRet.create(imageOri.size(), CV_8UC1);   // Filtered image
    imageExtention.create(imageOri.rows+2, imageOri.cols+2, CV_8UC1);

    /************************************************************************
    *
    * Expand outer boundary for original image
    *
    *************************************************************************/
    for(int y=0; y<imageExtention.rows; y++)
        for(int x=0; x<imageExtention.cols; x++)
        {
            if(x==0)//left;
            {
                if(y==0)//left top
                {
                    Mpixel(imageExtention, x, y) = Mpixel(imageOri, x, y);
                }
                else if(y==imageExtention.rows-1) //left bottom
                {
                    Mpixel(imageExtention, x, y) = Mpixel(imageOri, x, y-2);
                }
                else // left except left top and left bottom
                {
                    Mpixel(imageExtention, x, y) = Mpixel(imageOri, x, y-1);
                }
            }
            else if(x==imageExtention.cols-1)//right
            {
                if(y==0)//right top
                {
                    Mpixel(imageExtention, x, y) = Mpixel(imageOri, x-2, y);
                }
                else if(y==imageExtention.rows-1) //right bottom
                {
                    Mpixel(imageExtention, x, y) = Mpixel(imageOri, x-2, y-2);
                }
                else // right except right top and right bottom
                {
                    Mpixel(imageExtention, x, y) = Mpixel(imageOri, x-2, y-1);
                }
            }
            else if(y==0)//top
            {
                if(x!=0 && x!=imageExtention.cols-1) //top except left top and right top
                {
                    Mpixel(imageExtention, x, y) = Mpixel(imageOri, x-1, y);
                }
            }
            else if(y==imageExtention.rows-1)//bottom
            {
                if(x!=0 && x!=imageExtention.cols-1) //bottom except left bottom and right bottom
                {
                    Mpixel(imageExtention, x, y) = Mpixel(imageOri, x-1, y-2);
                }
            }
            else
                Mpixel(imageExtention, x, y) = Mpixel(imageOri, x-1, y-1);

        }

    /************************************************************************
    *
    * Read the original image and then create space for the two Mat being used
    * Output filtered image
    *
    *************************************************************************/
    for(int y=1; y<imageExtention.rows-1; y++)
        for(int x=1; x<imageExtention.cols-1; x++)
        {
            pMask[0] = Mpixel(imageExtention, x-1, y-1);
            pMask[1] = Mpixel(imageExtention, x, y-1);
            pMask[2] = Mpixel(imageExtention, x+1, y-1);
            pMask[3] = Mpixel(imageExtention, x-1, y);
            pMask[4] = Mpixel(imageExtention, x, y);
            pMask[5] = Mpixel(imageExtention, x+1, y);
            pMask[6] = Mpixel(imageExtention, x-1, y+1);
            pMask[7] = Mpixel(imageExtention, x, y+1);
            pMask[8] = Mpixel(imageExtention, x+1, y+1);

            qsort(pMask, 9, sizeof(int), compare_for_median);

            // Important part
            // Improve the accuracy
            if(pMask[4]<5)
                Mpixel(imageRet, x-1, y-1) = 0;
            else
                Mpixel(imageRet, x-1, y-1) = pMask[4];
        }


    return imageRet;
}


int count_blob(Mat imgOrigin)
{
    obj_vector vectorOfObj;   // The main data structure which controls the entire sperated objects
    int counter=-1, s1, s2;   //counter refers to the index of vectorOfObj.
                            // s1 and s2 are the auxiliary
    int matrixA[imgOrigin.cols+1][imgOrigin.rows+1] = {-1};        //This is a matrix to differiate the objects
    Mat imgOriginExtraBoundary;

    /************************************************************************
    *
    * Read the original image and then create space for the two Mat being used
    *
    *************************************************************************/
    imgOriginExtraBoundary.create(imgOrigin.rows+1, imgOrigin.cols+1, CV_8UC1);



    clock_t  clockBegin, clockEnd;
    clockBegin = clock();
    /************************************************************************
    *
    * Copy original matrix into imgOriginExtraBoundary and initialize matrixA -1
    *
    *************************************************************************/
    //print_mat(imgOriginExtraBoundary, "------extra boundary image-----");
    for(int y=0; y<imgOriginExtraBoundary.rows; y++)
        for(int x=0; x<imgOriginExtraBoundary.cols; x++)
        {
            if(x==0 || y==0 )
            {
                Mpixel(imgOriginExtraBoundary, x, y) = 0;
            }
            else
            {
                Mpixel(imgOriginExtraBoundary, x, y) = Mpixel(imgOrigin, x-1, y-1);
            }
            matrixA[x][y] = -1;
        }


    clockEnd = clock();
    printf("Expanding the origin matrix will take %ld ms\n", clockEnd - clockBegin);


    clockBegin = clock();
    /************************************************************************
    *
    * Implement the algorithm of Object Labelling using 4-adjacency
    *
    *************************************************************************/
    for(int y=1; y<imgOriginExtraBoundary.rows; y++)
        for(int x=1; x<imgOriginExtraBoundary.cols; x++)
        {
            if((int)Mpixel(imgOriginExtraBoundary, x, y)  != 0)
            {
                if(Mpixel(imgOriginExtraBoundary, x-1, y)!=0 || Mpixel(imgOriginExtraBoundary, x, y-1)!=0)
                {
                    s1 = matrixA[x-1][y];
                    s2 = matrixA[x][y-1];

                    if(s1 != -1)
                    {
                        point_set * setOfObj = &vectorOfObj[s1];
                        setOfObj->insert(Point_Oscar(x, y));
                        matrixA[x][y] = s1;
                    }

                    if(s2 != -1)
                    {
                        // Create a new set and insert a point which satisfies the condition
                        point_set * setOfObj = &vectorOfObj[s2];
                        setOfObj->insert(Point_Oscar(x, y));
                        matrixA[x][y] = s2;
                    }

                    if((s1 != s2) && (s1 != -1) && (s2 !=-1))
                    {
                        point_set * setOfObj1 = &vectorOfObj[s1];
                        point_set * setOfObj2 = &vectorOfObj[s2];
                        for(point_set::iterator it=setOfObj2->begin(); it!=setOfObj2->end(); it++)
                        {
                            setOfObj1->insert(*it);
                            int x = ((Point_Oscar)*it).x;
                            int y = ((Point_Oscar)*it).y;
                            matrixA[x][y] = s1;
                        }
                        setOfObj2->clear();
                        setOfObj2 = NULL;

                        g_counter--;
                    }

                }
                else
                {
                    counter++;
                    point_set setOfObj;
                    setOfObj.insert(Point_Oscar(x, y));
                    vectorOfObj.push_back(setOfObj);
                    matrixA[x][y] = counter;
                    g_counter++;
                }
            }

        }

    clockEnd = clock();
    printf("Algorithm will take %ld ms\n", clockEnd - clockBegin);

    int num = get_num_of_object(vectorOfObj);
    Mat imageColor = color_object(imgOrigin, vectorOfObj, num);
    char printit[100];
    sprintf(printit,"%d",num);
    putText(imageColor, printit, cvPoint(10,30), FONT_HERSHEY_PLAIN, 2, cvScalar(255,255,255), 2, 8);

    namedWindow("Figure4", 0);
    imshow("Figure4",imageColor);
    return num;
}

/************************************************************************
*
* Calculate the number of single objects.
*
*************************************************************************/
int get_num_of_object(obj_vector vectorOfObj)
{
    int num = 0;
    if(!vectorOfObj.empty())
    {
        for(obj_vector::iterator it=vectorOfObj.begin(); it!=vectorOfObj.end(); it++)
        {
            if( !((point_set)*it).empty() && ((point_set)*it).size() >30)
            {
                num++;
            }
        }
    }
    return num;
}

Mat color_object(Mat imageOri, obj_vector vectorOfObj, int numOfObj)
{
    Mat imageRet;
    imageRet.create(imageOri.size(), CV_8UC3);

    int obj_no = 0; // the variable for controling color
    int color_value = 255/numOfObj;

    /************************************************************************
    *
    * copy original image
    *
    *************************************************************************/

    for(int y=0; y<imageOri.rows; y++)
        for(int x=0; x<imageOri.cols; x++)
        {
            MpixelR(imageRet, x, y) = Mpixel(imageOri, x, y);
            MpixelG(imageRet, x, y) = Mpixel(imageOri, x, y);
            MpixelB(imageRet, x, y) = Mpixel(imageOri, x, y);
        }

    /************************************************************************
    *
    * check each unit of vector and color each unit according to the position of point in it
    *
    *************************************************************************/
    if(!vectorOfObj.empty())
    {
        for(obj_vector::iterator it=vectorOfObj.begin(); it!=vectorOfObj.end(); it++)
        {
            //if( !((point_set)*it).empty() && ((point_set)*it).size() >50)
            if( !((point_set)*it).empty()) // it may occur empty unit because two neighbour units will be merged into one of them, and another one will be emptied
            {
                obj_no++;
                int R_G_B[3] = {0};
                R_G_B[0] = color_value * obj_no;
                R_G_B[1] = color_value * (obj_no+3);
                R_G_B[2] = color_value * (obj_no+5);

                point_set setOfObj =  *it;
                for(point_set::iterator it= setOfObj.begin(); it!=setOfObj.end(); it++)
                {
                    // The reason about the operation of (x-1) is that
                    // the position of the points recorded in sets is one row and one column more
                    int x = ((Point_Oscar)*it).x-1;
                    int y = ((Point_Oscar)*it).y-1;

                    MpixelG(imageOri, x, y) = R_G_B[0];
                    MpixelB(imageOri, x, y) = R_G_B[1];
                    MpixelR(imageOri, x, y) = R_G_B[2];
                }
            }
        }
    }
    return imageOri;
}



Mat SmoothingFilter(Mat image, double* mask, int side)
{
    Mat imageRet, imageExtention;
    imageRet.create(image.size(), CV_8UC1);
    //imageExtention.create(image., CV_8UC1);
    imageExtention.create(image.rows+2, image.cols+2, CV_8UC1);

    for(int y=0; y<imageExtention.cols; y++)
        for(int x=0; x<imageExtention.rows; x++)
        {
            if(x==0)//left;
            {
                if(y==0)//left top
                {
                    imageExtention.at<uchar>(y,x) = image.at<uchar>(y, x);
                }
                else if(y==imageExtention.rows-1) //left bottom
                {
                    imageExtention.at<uchar>(y,x) = image.at<uchar>(y-2, x);
                }
                else // left except left top and left bottom
                {
                    imageExtention.at<uchar>(y,x) = image.at<uchar>(y-1, x);
                }
            }
            else if(x==imageExtention.cols-1)//right
            {
                if(y==0)//right top
                {
                    imageExtention.at<uchar>(y,x) = image.at<uchar>(y, x-2);
                }
                else if(y==imageExtention.rows-1) //right bottom
                {
                    imageExtention.at<uchar>(y,x) = image.at<uchar>(y-2, x-2);
                }
                else // right except right top and right bottom
                {
                    imageExtention.at<uchar>(y,x) = image.at<uchar>(y-1, x-2);
                }
            }
            else if(y==0)//top
            {
                if(x!=0 && x!=imageExtention.cols-1) //top except left top and right top
                {
                    imageExtention.at<uchar>(y,x) = image.at<uchar>(y, x-1);
                }
            }
            else if(y==imageExtention.rows-1)//bottom
            {
                if(x!=0 && x!=imageExtention.cols-1) //bottom except left bottom and right bottom
                {
                    imageExtention.at<uchar>(y,x) = image.at<uchar>(y-2, x-1);
                }
            }
            else
                imageExtention.at<uchar>(y,x) = image.at<uchar>(y-1, x-1);
        }

    for(int x=1; x<imageExtention.cols-side+1; x++)
        for(int y=1; y<imageExtention.rows-side+1; y++)
        {

            imageRet.at<uchar>(y-1, x-1) = mask[0] * (double)imageExtention.at<uchar>(y-1,x-1) +
                                            mask[1] * (double)imageExtention.at<uchar>(y-1,x) +
                                            mask[2] * (double)imageExtention.at<uchar>(y-1,x+1) +
                                            mask[3] * (double)imageExtention.at<uchar>(y,x-1) +
                                            mask[4] * (double)imageExtention.at<uchar>(y,x) +
                                            mask[5] * (double)imageExtention.at<uchar>(y,x+1) +
                                            mask[6] * (double)imageExtention.at<uchar>(y+1,x-1) +
                                            mask[7] * (double)imageExtention.at<uchar>(y+1,x) +
                                            mask[8] * (double)imageExtention.at<uchar>(y+1,x+1);
        }


    return imageRet;
}



