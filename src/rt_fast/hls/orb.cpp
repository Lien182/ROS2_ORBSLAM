#include "reconos_calls.h"
#include "reconos_thread.h"

#include <stdint.h>



#include "hls_video.h"


using namespace hls;

const int EDGE_THRESHOLD = 19;

const int iniThFAST = 20;
const int minThFAST = 7;

#define W 30

#define FAST_WINDOW_SIZE 50

#define IMAGE_CACHE_WIDTH   1280
#define IMAGE_CACHE_HEIGHT  128

//generate array 
template<int PSize,int KERNEL_SIZE,typename T, int N>
void __FAST_t_opr(
        uint8_t* _src,
        Point_<T>                    (&_keypoints)[N],
        HLS_TNAME(HLS_8UC1)                    _threshold,
        bool                    _nonmax_supression,
        int                     (&flag)[PSize][2], 
        int &nPoints
        )
{
    typedef typename pixel_op_type<HLS_TNAME(HLS_8UC1)>::T INPUT_T;
    LineBuffer<KERNEL_SIZE-1,FAST_WINDOW_SIZE,INPUT_T>    k_buf;
    LineBuffer<2,FAST_WINDOW_SIZE+KERNEL_SIZE,ap_int<16> >         core_buf;
    Window<3,3,ap_int<16> >                            core_win;
    Window<KERNEL_SIZE,KERNEL_SIZE,INPUT_T>       win;
    Scalar<HLS_MAT_CN(HLS_8UC1), HLS_TNAME(HLS_8UC1)>             s;
    int rows= FAST_WINDOW_SIZE;
    int cols= FAST_WINDOW_SIZE;
    assert(rows <= FAST_WINDOW_SIZE);
    assert(cols <= FAST_WINDOW_SIZE);
    int kernel_half=KERNEL_SIZE/2;
    ap_uint<2> flag_val[PSize+PSize/2+1];
    int  flag_d[PSize+PSize/2+1];
#pragma HLS ARRAY_PARTITION variable=flag_val dim=0
#pragma HLS ARRAY_PARTITION variable=flag_d dim=0
    int index=0;
    int offset=KERNEL_SIZE/2;

    if(_nonmax_supression)
    {
        offset=offset+1;
    }
 loop_height: for(HLS_SIZE_T i=0;i<rows+offset;i++) {
    loop_width: for(HLS_SIZE_T j=0;j<cols+offset;j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
            if(i<rows&&j<cols) {
                for(int r= 0;r<KERNEL_SIZE;r++) {
                    for(int c=0;c<KERNEL_SIZE-1;c++) {
                        win.val[r][c]=win.val[r][c+1];//column left shift
                    }
                }
                win.val[0][KERNEL_SIZE-1]=k_buf.val[0][j];
                for(int buf_row= 1;buf_row< KERNEL_SIZE-1;buf_row++) {
                    win.val[buf_row][KERNEL_SIZE-1]=k_buf.val[buf_row][j];
                    k_buf.val[buf_row-1][j]=k_buf.val[buf_row][j];
                }
                //-------
                s = _src[i*50+j];
                win.val[KERNEL_SIZE-1][KERNEL_SIZE-1]=s.val[0];
                k_buf.val[KERNEL_SIZE-2][j]=s.val[0];
            }
            //------core
            for(int r= 0;r<3;r++)
            {
                for(int c=0;c<3-1;c++)
                {
                    core_win.val[r][c]=core_win.val[r][c+1];//column left shift
                }
            }
            core_win.val[0][3-1]=core_buf.val[0][j];
            for(int buf_row= 1;buf_row< 3-1;buf_row++)
            {
                core_win.val[buf_row][3-1]=core_buf.val[buf_row][j];
                core_buf.val[buf_row-1][j]=core_buf.val[buf_row][j];
            }
            int core=0;
            //output
            //if(i>=KERNEL_SIZE-1&&j>=KERNEL_SIZE-1)
            if(i>=KERNEL_SIZE-1 && i<rows && j>=KERNEL_SIZE-1 & j<cols)
            {
                //process
                bool iscorner=fast_judge<PSize>(win,(INPUT_T)_threshold,flag_val,flag_d,flag,core,_nonmax_supression);
                if(iscorner&&!_nonmax_supression)
                {
                    if(index<N)
                    {
                    _keypoints[index].x=j-offset;
                    _keypoints[index].y=i-offset;
                    index++;
                    }
                }
            }
            if(i>=rows||j>=cols)
            {
                core=0;
            }
            if(_nonmax_supression)
            {
                core_win.val[3-1][3-1]=core;
                core_buf.val[3-2][j]=core;
                if(i>=KERNEL_SIZE&&j>=KERNEL_SIZE&&core_win.val[1][1]!=0)
                {
                    bool iscorner=fast_nonmax(core_win);
                    if(iscorner)
                    {
                    if(index<N)
                    {
                        _keypoints[index].x=j-offset;
                        _keypoints[index].y=i-offset;
                        index++;
                    }
                    }
                }
            }

        }
    }

	nPoints = index;
}


//generate array 
template<int PSize,int KERNEL_SIZE,typename T, int N, int SRC_T,int ROWS,int COLS>
void _FAST_t_opr(
        hls::Mat<ROWS,COLS,SRC_T>    &_src,
        Point_<T>                    (&_keypoints)[N],
        HLS_TNAME(SRC_T)                    _threshold,
        bool                    _nonmax_supression,
        int                     (&flag)[PSize][2], 
        int &nPoints
        )
{
    typedef typename pixel_op_type<HLS_TNAME(SRC_T)>::T INPUT_T;
    LineBuffer<KERNEL_SIZE-1,COLS,INPUT_T>    k_buf;
    LineBuffer<2,COLS+KERNEL_SIZE,ap_int<16> >         core_buf;
    Window<3,3,ap_int<16> >                            core_win;
    Window<KERNEL_SIZE,KERNEL_SIZE,INPUT_T>       win;
    Scalar<HLS_MAT_CN(SRC_T), HLS_TNAME(SRC_T)>             s;
    int rows= _src.rows;
    int cols= _src.cols;
    assert(rows <= ROWS);
    assert(cols <= COLS);
    int kernel_half=KERNEL_SIZE/2;
    ap_uint<2> flag_val[PSize+PSize/2+1];
    int  flag_d[PSize+PSize/2+1];
#pragma HLS ARRAY_PARTITION variable=flag_val dim=0
#pragma HLS ARRAY_PARTITION variable=flag_d dim=0
    int index=0;
    int offset=KERNEL_SIZE/2;

    if(_nonmax_supression)
    {
        offset=offset+1;
    }
 loop_height: for(HLS_SIZE_T i=0;i<rows+offset;i++) {
    loop_width: for(HLS_SIZE_T j=0;j<cols+offset;j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
            if(i<rows&&j<cols) {
                for(int r= 0;r<KERNEL_SIZE;r++) {
                    for(int c=0;c<KERNEL_SIZE-1;c++) {
                        win.val[r][c]=win.val[r][c+1];//column left shift
                    }
                }
                win.val[0][KERNEL_SIZE-1]=k_buf.val[0][j];
                for(int buf_row= 1;buf_row< KERNEL_SIZE-1;buf_row++) {
                    win.val[buf_row][KERNEL_SIZE-1]=k_buf.val[buf_row][j];
                    k_buf.val[buf_row-1][j]=k_buf.val[buf_row][j];
                }
                //-------
                _src>>s;
                win.val[KERNEL_SIZE-1][KERNEL_SIZE-1]=s.val[0];
                k_buf.val[KERNEL_SIZE-2][j]=s.val[0];
            }
            //------core
            for(int r= 0;r<3;r++)
            {
                for(int c=0;c<3-1;c++)
                {
                    core_win.val[r][c]=core_win.val[r][c+1];//column left shift
                }
            }
            core_win.val[0][3-1]=core_buf.val[0][j];
            for(int buf_row= 1;buf_row< 3-1;buf_row++)
            {
                core_win.val[buf_row][3-1]=core_buf.val[buf_row][j];
                core_buf.val[buf_row-1][j]=core_buf.val[buf_row][j];
            }
            int core=0;
            //output
            //if(i>=KERNEL_SIZE-1&&j>=KERNEL_SIZE-1)
            if(i>=KERNEL_SIZE-1 && i<rows && j>=KERNEL_SIZE-1 & j<cols)
            {
                //process
                bool iscorner=fast_judge<PSize>(win,(INPUT_T)_threshold,flag_val,flag_d,flag,core,_nonmax_supression);
                if(iscorner&&!_nonmax_supression)
                {
                    if(index<N)
                    {
                    _keypoints[index].x=j-offset;
                    _keypoints[index].y=i-offset;
                    index++;
                    }
                }
            }
            if(i>=rows||j>=cols)
            {
                core=0;
            }
            if(_nonmax_supression)
            {
                core_win.val[3-1][3-1]=core;
                core_buf.val[3-2][j]=core;
                if(i>=KERNEL_SIZE&&j>=KERNEL_SIZE&&core_win.val[1][1]!=0)
                {
                    bool iscorner=fast_nonmax(core_win);
                    if(iscorner)
                    {
                        if(index<N)
                        {
                            _keypoints[index].x=j-offset;
                            _keypoints[index].y=i-offset;
                            index++;
                        }
                    }
                }
            }

        }
    }

	nPoints = index;
}

template<typename T, int N, int SRC_T,int ROWS,int COLS>
void  _FASTX(
        Mat<ROWS,COLS,SRC_T>    &_src,
        Point_<T> (&_keypoints)[N],
        HLS_TNAME(SRC_T)    _threshold,
        bool   _nomax_supression,
        int &_nPoints
        )
{
#pragma HLS INLINE
    int flag[16][2]={{3,0},{4,0},{5,1},{6,2},{6,3},{6,4},{5,5},{4,6},
        {3,6},{2,6},{1,5},{0,4},{0,3},{0,2},{1,1},{2,0}};
    _FAST_t_opr<16,7>(_src,_keypoints,_threshold,_nomax_supression,flag,_nPoints);
}

template<typename T, int N>
void  __FASTX(
        uint8_t * _src,
        Point_<T> (&_keypoints)[N],
        HLS_TNAME(HLS_8UC1)    _threshold,
        bool   _nomax_supression,
        int &_nPoints
        )
{
#pragma HLS INLINE
    int flag[16][2]={{3,0},{4,0},{5,1},{6,2},{6,3},{6,4},{5,5},{4,6},
        {3,6},{2,6},{1,5},{0,4},{0,3},{0,2},{1,1},{2,0}};
    __FAST_t_opr<16,7>(_src,_keypoints,_threshold,_nomax_supression,flag,_nPoints);
}


void mat_copy( uint8 * data, hls::Mat<FAST_WINDOW_SIZE,FAST_WINDOW_SIZE,HLS_8UC1>  &_dest, int iniY, int iniX, int cache_cnt, uint32 offset)
{
    for(int i = 0; i < (FAST_WINDOW_SIZE); i++)
    {
        for(int j = 0; j < (FAST_WINDOW_SIZE); j++)
        {
            _dest.write(data[(j+iniY)+ ((iniX+i+cache_cnt) %IMAGE_CACHE_HEIGHT) *IMAGE_CACHE_WIDTH]);
        }
    }
}

void array_copy( uint8 * data, uint8_t * _dest, int iniY, int iniX, int cache_cnt, uint32 offset)
{
    for(int i = 0; i < (FAST_WINDOW_SIZE); i++)
    {
        for(int j = 0; j < (FAST_WINDOW_SIZE); j++)
        {
            _dest[j+i*50] = data[offset + (j+iniY)+ ((iniX+i+cache_cnt) %IMAGE_CACHE_HEIGHT) *IMAGE_CACHE_WIDTH];
        }
    }
}


#define read_next_lines {for(int i = 0; i < FAST_WINDOW_SIZE; i++){ \
                                    offset = (image_ptr + cache_cnt*(image_width))&3; \
                                    MEM_READ(((image_ptr + cache_cnt*(image_width))&(~3)), image_data + IMAGE_CACHE_WIDTH*(cache_cnt%IMAGE_CACHE_HEIGHT),\
                                    ((image_width+offset+3)&(~3)));} cache_cnt+=50;}



THREAD_ENTRY() {

    THREAD_INIT();
	//uint32 initdata = GET_INIT_DATA();

    uint8 image_data[IMAGE_CACHE_WIDTH*IMAGE_CACHE_HEIGHT];

	while(1)
	{
        uint32 cache_cnt =0 ;
       

		uint32 image_ptr    = MBOX_GET(resources_fast_request);
		uint32 image_width  = MBOX_GET(resources_fast_request);
		uint32 image_height = MBOX_GET(resources_fast_request);

        uint32 offset = (image_ptr + cache_cnt*image_width) & 3;
        
		const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = image_width -EDGE_THRESHOLD+3;
        const int maxBorderY = image_height-EDGE_THRESHOLD+3;

        const int width = (maxBorderX-minBorderX);
        const int height = (maxBorderY-minBorderY);

        const int wCell = 50;
        const int hCell = 50;
        const int nCols = width/50;
        const int nRows = height/50;
        
        //read_next_lines;



		for(int i=0; i<nRows; i++)
		{           

            read_next_lines;
           

			const int iniY =minBorderY+i*hCell; //this was float
			int maxY = iniY+hCell+6; //this was float

			if(iniY>=maxBorderY-3)
				continue;
			if(maxY>maxBorderY)
				maxY = maxBorderY;

			for(int j=0; j<nCols; j++)
			{
				const int iniX =minBorderX+j*wCell; //this was float
				int maxX = iniX+wCell+6;//this was float
				if(iniX>=maxBorderX-6)
					continue;
				if(maxX>maxBorderX)
					maxX = maxBorderX;

				hls::Point_<uint16> vKeysCell[256];
                int nPoints;

                hls::Mat<FAST_WINDOW_SIZE,FAST_WINDOW_SIZE,HLS_8UC1> cell_stream(FAST_WINDOW_SIZE,FAST_WINDOW_SIZE);
                //uint8_t cell_array[50*50];
                

                //#pragma HLS stream depth=200 variable=cell_stream.data_stream
                {   
                    #pragma HLS dataflow
                    #pragma HLS stream depth=2500 variable=cell_stream.data_stream
                    
                    mat_copy( image_data, cell_stream, iniY,  iniX,  cache_cnt, offset);
				    //array_copy( image_data, cell_array, iniY,  iniX,  cache_cnt);
				    
                    _FASTX(cell_stream, vKeysCell,  iniThFAST, true, nPoints);
                    //for(int i = 0; i < 100; i+=2)
                    //MBOX_PUT(resources_fast_response, (uint32)image_data[0]);
                }
	
               /*
				if(nPoints == 0)
				{
                   
                    //#pragma HLS stream depth=200 variable=cell_stream.data_stream
                    {
                        #pragma HLS dataflow
                        #pragma HLS INLINE
                        mat_copy( image_data, cell_stream, iniY,  iniX,  image_width);
                        //FPGA::FPGA_FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),	vKeysCell,minThFAST,true);
                        nPoints = _FASTX<uint16, 256,HLS_8UC1, FAST_WINDOW_SIZE,FAST_WINDOW_SIZE>( cell_stream, vKeysCell,  iniThFAST, true);
                    }
				}
                
                */

                for(int i = 0; i < nPoints; i++)
                    MBOX_PUT(resources_fast_response, vKeysCell[i].x | (vKeysCell[i].y << 16));

				
			}
		}
        
        MBOX_PUT(resources_fast_response, 0xffffffff);
         
	}

}