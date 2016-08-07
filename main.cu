#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv\cv.h>

#include <cuda.h>
using namespace cv;
using namespace std;
// funciones  get y set
int getPixel(Mat& mat, int fil, int col){ return (uchar)mat.at<uchar>(fil, col); }
cv::Vec3b& getPixel3(Mat& mat, int j, int i){ return mat.at<cv::Vec3b>(j, i); }
void setPixel3(Mat& mat, int fil, int col, cv::Vec3b& color){ mat.at<cv::Vec3b>(fil, col) = color; }
void setPixel1(Mat& mat, int fil, int col, int pixel){ mat.at<uchar>(fil, col) = pixel; }
void ColorReducer(Mat& frame, Mat& output, int factor){

	int nl = frame.rows;
	int nc = frame.cols * frame.channels();

	uchar* data = frame.ptr<uchar>(0);
	for (int i = 0; i<nc*nl; i++) {
		data[i] = data[i] / factor*factor + factor / 2;
	}
	output = frame.clone();
}


void print(int *mat, int N){

	int i, j;
	cout << endl;
	for (i = 0; i < N; i++){			// load arrays with some numbers
		for (j = 0; j < N; j++) {
			cout << mat[i * N + j] << " ";
		}cout << endl;
	}
}


__global__ void gpu_matrixadd(uchar *c, int N) {

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int index  = row * N + col;
	int k = 1;
	int xIndex2 = (row-k)* N + (col-k);
	int xIndex3 = (row - k) * N + col;
	int xIndex4 = (row - k) * N + (col + k);
	int xIndex5 = (row + k)* N + (col - k);
	int xIndex6 = (row + k)* N + (col);
	int xIndex7 = (row + k)* N + (col + k);

	int yIndex2 = (row - k)* N + (col - k);
	int yIndex3 = (row) * N + (col-k);
	int yIndex4 = (row + k) * N + (col - k);
	int yIndex5 = (row - k) * N + (col + k);
	int yIndex6 = (row)* N + (col+k);
	int yIndex7 = (row + k)* N + (col + k);

	int xVariation = 0, yVariation = 0, sum = 0;
	int factor = 2;
	int threshold = 10;
	if (col < N && row < N){
		//c[index] = (c[index] > 125) ? 0 : 255;
		xVariation = c[xIndex2] + factor* c[xIndex3] + c[xIndex4] - c[xIndex5] - factor * c[xIndex6] -c[xIndex7];
		yVariation = c[yIndex2] + factor* c[yIndex3] + c[yIndex4] - c[yIndex5] - factor * c[yIndex6] -c[yIndex7];
		sum = sqrt(double(xVariation*xVariation + yVariation*yVariation));  
		//sum = abs(xVariation) + abs(yVariation);
		sum = sum > 255 ? 255 : sum;
		sum = sum < 0 ? 0 : sum;
		c[index] = sum;
		//sum = uchar(sum);

		//c[index] = (sum > threshold) ? 255 : 0;
		
	}

}


//CPU METHODS



// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGradient(Mat image, int x, int y)
{
	return image.at<uchar>(y - 1, x - 1) +
		2 * image.at<uchar>(y - 1, x) +
		image.at<uchar>(y - 1, x + 1) -
		image.at<uchar>(y + 1, x - 1) -
		2 * image.at<uchar>(y + 1, x) -
		image.at<uchar>(y + 1, x + 1);
}
// Computes the x component of the gradient vector
// at a given point in a image.
// returns gradient in the x direction
int xGradient(Mat image, int x, int y)
{
	return image.at<uchar>(y - 1, x - 1) +
		2 * image.at<uchar>(y, x - 1) +
		image.at<uchar>(y + 1, x - 1) -
		image.at<uchar>(y - 1, x + 1) -
		2 * image.at<uchar>(y, x + 1) -
		image.at<uchar>(y + 1, x + 1);
}


void cpu_edgeDetection(Mat& src){
	
	Mat dst;
	dst = src.clone();
	dst.create(src.rows, src.cols, src.type());
	/*
	for (int y = 0; y < src.rows; y++)
	for (int x = 0; x < src.cols; x++)
	dst.at<uchar>(y, x) = 0.0;

	*/
	int gx, gy, sum;

	for (int y = 1; y < src.rows - 1; y++){
		for (int x = 1; x < src.cols - 1; x++){
			gx = xGradient(src, x, y);
			gy = yGradient(src, x, y);
			sum = abs(gx) + abs(gy);
			sum = sum > 255 ? 255 : sum;
			sum = sum < 0 ? 0 : sum;
			dst.at<uchar>(y, x) = sum;
		}
	}
	for (int y = 1; y < src.rows - 1; y++){
		for (int x = 1; x < src.cols - 1; x++){
			dst.at<uchar>(y, x) = (dst.at<uchar>(y, x) <100) ? 0 : 255;
		}
	}
	imwrite("cpu_result.jpg", dst);
}

int main(int argc, char *argv[])  {

	Mat img;
	Mat outimg;
	img = imread("dragon4096.png", 0);
	imwrite("in.jpg", img);
	GaussianBlur(img, img, Size(7, 7), 0, 0);
	cout << img.rows << "-" << img.cols << "-" << img.type() << endl;
	outimg.create(img.rows, img.cols, img.type());

	int i, j; 					// loop counters
	int Grid_Dim_x = 128, Grid_Dim_y = 128;			//Grid structure values
	int Block_Dim_x = 32, Block_Dim_y = 32;		//Block structure values

	int noThreads_x, noThreads_y;		// number of threads available in device, each dimension
	int noThreads_block;				// number of threads in a block

	int N = 4096;  					// size of array in each dimension
	uchar *c;
	uchar *dev_c;
	int size;					// number of bytes in arrays

	cudaEvent_t start, stop;     		// using cuda events to measure time
	float elapsed_time_ms;       		// which is applicable for asynchronous code also

	/* --------------------ENTER INPUT PARAMETERS AND DATA -----------------------*/



	printf("Device characteristics -- some limitations (compute capability 1.0)\n");
	printf("		Maximum number of threads per block = 512\n");
	printf("		Maximum sizes of x- and y- dimension of thread block = 512\n");
	printf("		Maximum size of each dimension of grid of thread blocks = 65535\n");

	printf("Enter size of array in one dimension (square array), currently %d\n", N);
	scanf("%d", &N);

	do {
		printf("\nEnter number of blocks per grid in x dimension), currently %d  : ", Grid_Dim_x);
		scanf("%d", &Grid_Dim_x);

		printf("\nEnter number of blocks per grid in y dimension), currently %d  : ", Grid_Dim_y);
		scanf("%d", &Grid_Dim_y);

		printf("\nEnter number of threads per block in x dimension), currently %d  : ", Block_Dim_x);
		scanf("%d", &Block_Dim_x);

		printf("\nEnter number of threads per block in y dimension), currently %d  : ", Block_Dim_y);
		scanf("%d", &Block_Dim_y);

		noThreads_x = Grid_Dim_x * Block_Dim_x;		// number of threads in x dimension
		noThreads_y = Grid_Dim_y * Block_Dim_y;		// number of threads in y dimension

		noThreads_block = Block_Dim_x * Block_Dim_y;	// number of threads in a block

		if (noThreads_x < N) printf("Error -- number of threads in x dimension less than number of elements in arrays, try again\n");
		else if (noThreads_y < N) printf("Error -- number of threads in y dimension less than number of elements in arrays, try again\n");
		else if (noThreads_block > 1024) printf("Error -- too many threads in block, try again\n");
		else printf("Number of threads not used = %d\n", noThreads_x * noThreads_y - N * N);

	} while (noThreads_x < N || noThreads_y < N || noThreads_block > 1024);

	dim3 Grid(Grid_Dim_x, Grid_Dim_x);		//Grid structure
	dim3 Block(Block_Dim_x, Block_Dim_y);	//Block structure, threads/block limited by specific device

	size = N * N * sizeof(int);		// number of bytes in total in arrays

	//a = (int*)malloc(size);		//this time use dynamically allocated memory for arrays on host
	//b = (int*)malloc(size);
	c = (uchar*)malloc(size);		// results from GPU
	//d = (int*)malloc(size);		// results from CPU

	for (i = 0; i < N; i++)			// load arrays with some numbers
		for (j = 0; j < N; j++) {
			c[i * N + j] = getPixel(img, i, j);
		}

	/* ------------- COMPUTATION DONE ON GPU ----------------------------*/

	cudaMalloc((void**)&dev_c, size);
	cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//	cudaEventSynchronize(start);  	// Needed?
	gpu_matrixadd << <Grid, Block >> >(dev_c, N);
	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);  // print out execution time

	/* ------------- COMPUTATION DONE ON HOST CPU ----------------------------*/

	cudaEventRecord(start, 0);		// use same timing
	//	cudaEventSynchronize(start);  	// Needed?

	cpu_edgeDetection(img);

	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);


	printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms);  // print out execution time

	
	int pixel = 0;
	for (i = 0; i < N; i++)			
		for (j = 0; j < N; j++) {
			pixel = c[i * N + j];
			setPixel1(outimg, i, j, pixel);
		}
	
	imwrite("gpu_result.jpg", outimg);

	/* --------------  clean up  ---------------------------------------*/

		free(c);
		cudaFree(dev_c);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);


	system("pause");
	return 0;
}
