#include "FPGA.h"

#include <iostream>

#include <stdint.h>
#include <string.h>
#include <vector>

#define FAST_WINDOW_SIZE 50


#define MEM_READ(src, dest, n) memcpy((void*)dest, (void*)src, n)

#define read_next_lines {for(int i = 0; i < FAST_WINDOW_SIZE; i++){ MEM_READ(image_ptr + cache_cnt*image_width, image_data + IMAGE_CACHE_WIDTH*(cache_cnt%IMAGE_CACHE_HEIGHT), image_width );} cache_cnt+=50;}
#define W 30

#define IMAGE_CACHE_WIDTH   1400
#define IMAGE_CACHE_HEIGHT  128

void FPGA::FPGA_FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression )
{

    FAST(image, keypoints, threshold, nonmaxSuppression );
}


static const int EDGE_THRESHOLD = 19;
static const int iniThFAST = 20;
static const int minThFAST = 7;



void FPGA::Compute_Keypoints( uint8_t* image_ptr, uint32_t image_width, uint32_t image_height, uint32_t nfeatures, vector<uint32_t> & keypoints )
{
    uint8_t image_data[IMAGE_CACHE_WIDTH*IMAGE_CACHE_HEIGHT];

    uint32_t cache_cnt = 0;

    std::cout << "Image width: " << image_width << "; Image height " << image_height << std::endl;

    const int minBorderX = EDGE_THRESHOLD-3;
    const int minBorderY = minBorderX;
    const int maxBorderX = image_width -EDGE_THRESHOLD+3;
    const int maxBorderY = image_height-EDGE_THRESHOLD+3;

    const int width = (maxBorderX-minBorderX);
    const int height = (maxBorderY-minBorderY);

/*
    const int nCols = width/W;
    const int nRows = height/W;
    const int wCell = (width/nCols);
    const int hCell = (height/nRows);
*/

    const int wCell = 50;
    const int hCell = 50;
    const int nCols = width/50;
    const int nRows = height/50;


    read_next_lines;


    //std::cout << "nCols " << nCols << "; nRows " << nRows << "; wCell " << wCell << ";hCell " << hCell << std::endl;

    for(int i=0; i<nRows; i++)
    {

        read_next_lines;

        const int iniY =minBorderY+i*hCell;
        int maxY = iniY+hCell+6;

        if(iniY>=maxBorderY-3)
            continue;
        if(maxY>maxBorderY)
            maxY = maxBorderY;

        for(int j=0; j<nCols; j++)
        {
            const int iniX =minBorderX+j*wCell;
            int maxX = iniX+wCell+6;
            if(iniX>=maxBorderX-6)
                continue;
            if(maxX>maxBorderX)
                maxX = maxBorderX;


            

            /*
            cv::Mat * img = (cv::Mat *)image; 
            cv::Mat tmp = img->rowRange(iniY,iniY+50).colRange(iniX,iniX+50);
            */

            cv::Mat tmp = Mat(50, 50, CV_8UC1);

            for(int i = 0; i < (50); i++)
            {
                for(int j = 0; j < (50); j++)
                {
                    tmp.data[i*50+j] = image_data[(j+iniY)+ ((iniX+i+cache_cnt) %IMAGE_CACHE_HEIGHT) *IMAGE_CACHE_WIDTH];
                }
            }
            



            vector<cv::KeyPoint> vKeysCell;
            FPGA::FPGA_FAST(tmp, vKeysCell,minThFAST,true);
            //std::cout << "1. Attempt: Number of keypoints " << vKeysCell.size() << std::endl;

            //if(vKeysCell.empty())
            //{
            //    FPGA::FPGA_FAST(tmp, vKeysCell,minThFAST,true);
            //    std::cout << "2. Attempt: Number of keypoints " << vKeysCell.size() << std::endl;
            //}

            if(!vKeysCell.empty())
            {
                for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                {
                    (*vit).pt.x+=j*wCell;
                    (*vit).pt.y+=i*hCell;
                    keypoints.push_back((uint32_t)(*vit).pt.x | ((uint32_t)(*vit).pt.y << 16));
                }
            }

        }
    }
}
