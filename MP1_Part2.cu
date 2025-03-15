
// tristan guevarra - 20353857 - 21tag10
#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>

//cpu matrix multiplication (executed sequentially on the cpu)
void matrixMultiplicationCPU(float* P, float* M, float* N, int sizeN) {
    //iterating over each row
    for (int row = 0; row < sizeN; row++) { //iterating through
        //iterating over each column
        for (int col = 0; col < sizeN; col++) { //iterating through 2
            float sum = 0.0; //initializing the sum
            //iterating over elements of the row/M and column/N 
            for (int i = 0; i < sizeN; i++) { //iterating through 3
                sum += M[row * sizeN + i] * N[i * sizeN + col];
            }
            P[row * sizeN + col] = sum; //storing the computed value
        }
    }

}

//gpu matrix multiplication kernel (executed on the gpu with a single block and a single thread)
__global__ void kernelMatrixMultiplicationGPU(float* P, float* M, float* N, int sizeN) {
    //part2
    //kernel
    //iterating over each row
 //   for (int row = 0; row < sizeN; row++) { //iterating loop 1
 //       //iterating over each column
 //       for (int col = 0; col < sizeN; col++) { //iterating for loop 2
 //           float sum = 0.0f; //initializing sum 
 //           //iterating over elements of the row/M and column/N 
 //           for (int k = 0; k < sizeN; k++) { //iterating for loop 3
 //               sum += M[row * sizeN + k] * N[k * sizeN + col];
 //           }
 //          P[row * sizeN + col] = sum; //storing the computed value
 //       }
 //   }

    //part2-3
    //getting the row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y; //row
    int col = blockIdx.x * blockDim.x + threadIdx.x; //column

    //ensuring the bounds of the matrix, so making sure we are within
    if (row < sizeN && col < sizeN) { // if statement declaration
        float sum = 0.0f; //initializing sum
        for (int k = 0; k < sizeN; k++) { //for loop iterating from 0 -> size
            sum += M[row * sizeN + k] * N[k * sizeN + col];
        }
        P[row * sizeN + col] = sum; //storing result
    }

}


//function to measure the transfer time
void measuringTheTransferTimePart2_1() { //initializing function

    //defining an array
    int s[] = { 256,512,1024,2048,4096 }; //different matrix sizes/numbers "{256,512,1024,2048,4096}"
    //calculating the number of matrix sizes
    int ns = sizeof(s) / sizeof(int);

    for (int i = 0; i < ns; i++) { //initializing a for loop to interate through number of matrix sizes
        int size = s[i]; //current size
        int me = size * size; //matrix elements
        int mb = me * sizeof(float); //matrix bytes

        //allocating the memory on the host/cpu for two matrices
        float* host_M = (float*)malloc(mb);
        float* host_N = (float*)malloc(mb);
        float* device_M, * device_N; //declaring pointer for device/gpu memory

        //allocating the memory on the device/gpu
        cudaMalloc(&device_M, mb);
        cudaMalloc(&device_N, mb);
        //creating events for times
        cudaEvent_t start, stop;
        cudaEventCreate(&start); //initializing start through cudaEventCreate
        cudaEventCreate(&stop); //initializing stop through cudaEventCreate
        float hdTime, dhTime; //initializing variables hd and dh to store times
        //measuring the time to move data points from host/cpu to device/gpu

        cudaEventRecord(start); //starts timer
        //copying all host matrix to device
        cudaMemcpy(device_M, host_M, mb, cudaMemcpyHostToDevice); //host cpu copy matrix M to gpu
        cudaMemcpy(device_N, host_N, mb, cudaMemcpyHostToDevice); //device gpu copy matrix N to gpu

        cudaEventRecord(stop); //stops timer
        //used to ensure the event completes before measuring
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&hdTime, start, stop);

        cudaEventRecord(start);

        cudaMemcpy(host_M, device_M, mb, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_N, device_N, mb, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dhTime, start, stop);

        //printing the results from the current matrix size and time
        printf("size: %4d x %4d host->device: %.3f ms device->host:%.3f ms\n", size, size, hdTime, dhTime);
        //free allocated memory on the host/cpu
        free(host_M);
        free(host_N);
        //free allocated memory on the device/gpu
        cudaFree(device_M);
        cudaFree(device_N);
        //destroying cuda events to free resources
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() { //main function to execute the transfer time

    //part2-1
    measuringTheTransferTimePart2_1(); //calling the measuringTheTransferTimePart2_1() function
    //below: machine problem 1 part 2 part 2 -
    //define an array of matrix sizes to test (256x256, 512x512, 1024x1024)
    int p2sizes[] = { 256, 512, 1024 };
    //calculating the number of matrix sizes in the array
    int p2num = sizeof(p2sizes) / sizeof(int);
    //looping through different matrix sizes to perform matrix multiplication
    for (int i = 0; i < p2num; i++) {
        int sizeN = p2sizes[i]; //current matrix size
        int p2m = sizeN * sizeN * sizeof(float); //calculating the memory required for each matrix

        //printing current matrix size being processed "matrix size processed 256, 512, 1024"
        printf("matrix size processed 256, 512, 1024: %d x %d\n", sizeN, sizeN);

        //allocating host memory for matrices M, N, and output matrices for CPU and GPU
        float* hosting_M = (float*)malloc(p2m); //host matrix M
        float* hosting_N = (float*)malloc(p2m); //host matrix N
        float* p2hPcpu = (float*)malloc(p2m);  //host output matrix for CPU  
        float* p2gpu = (float*)malloc(p2m);    //host output matrix for GPU  

        //initializing matrices 
        for (int j = 0; j < sizeN * sizeN; j++) {
            hosting_M[j] = (float)(rand()) / RAND_MAX; //random value for M
            hosting_N[j] = (float)(rand()) / RAND_MAX; //random value for N
        }
        float* dM, * dN, * dP; //declaring device pointers for matrices M, N, and P

        //allocating memory on the device/GPU for matrices
        cudaMalloc(&dM, p2m);
        cudaMalloc(&dN, p2m);
        cudaMalloc(&dP, p2m);

        //copy input matrices from host/CPU to device/GPU
        cudaMemcpy(dM, hosting_M, p2m, cudaMemcpyHostToDevice);
        cudaMemcpy(dN, hosting_N, p2m, cudaMemcpyHostToDevice);


        //declaring CUDA events for GPU execution timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float tGPU, tCPU; //variable to store GPU execution time

        //define execution configuration (single block, single thread)
        //measuring GPU execution time 
        cudaEventRecord(start); //starting GPU timer
        kernelMatrixMultiplicationGPU << <1, 1 >> > (dP, dM, dN, sizeN); //launching kernel grid and block
        cudaEventRecord(stop); //stoping GPU timer
        cudaEventSynchronize(stop); //ensuring kernel execution is completed
        cudaEventElapsedTime(&tGPU, start, stop); //calculating the elapsed time 

        //copying the result matrix from device/GPU to host/CPU for verification
        cudaMemcpy(p2gpu, dP, p2m, cudaMemcpyDeviceToHost);

        //printing the execution times for CPU and GPU times computitions
        printf("CPU time: %.2f ms\n", 
               tCPU); //cpu times
        printf("GPU time: %.2f ms\n", 
               tGPU); //gpu times

        //allocation memory on host
        free(hosting_M);
        free(hosting_N);
        free(p2hPcpu);
        free(p2gpu);
        //allocation memory on device
        cudaFree(dM);
        cudaFree(dN);
        cudaFree(dP);
        //destroying CUDA start and stop
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        //part2-3
        int mS[] = { 256,512,1024,2048,4096 }; //matrix sizes to test
        int blockS[] = { 2,4,8,16,32 }; //block widths to experiment with
        //initializing for loop to iterate through the 5 aspects of matric and block sizes
        for (int i = 0; i < 5; i++) { //i could use sizeof like other parts but this makes it easier
            int sizeN = mS[i];
            int mS2 = sizeN * sizeN * sizeof(float); //calculating size
            //allocate host memory for matrices
            float* hoM = (float*)malloc(mS2);
            float* hoN = (float*)malloc(mS2);
            float* hoP = (float*)malloc(mS2);
            for (int j = 0; j < sizeN * sizeN; j++) { ///initialize matrices with random values
                hoM[j] = (float)(rand()) / RAND_MAX;
                hoN[j] = (float)(rand()) / RAND_MAX;
            }
            float* deOM, * deON, * deOP; //putting and allocating memory using pt devices
            cudaMalloc(&deOM, mS2); //device with size
            cudaMalloc(&deON, mS2);
            cudaMalloc(&deOP, mS2);
            cudaEvent_t start, stop; //event cuda
            cudaEventCreate(&start); //initializing start through cudaEventCreate
            cudaEventCreate(&stop); //initializing stop through cudaEventCreate
            float computation, dhT; //also from previous parts, it is the computation time and device to hosting
            cudaMemcpy(deOM, hoM, mS2, cudaMemcpyHostToDevice);
            cudaMemcpy(deON, hoN, mS2, cudaMemcpyHostToDevice);

            printf("the matrix size: %d\n", sizeN);
            //looping and iterating through different sizes of the block
            for (int widthb = 0; widthb < 5; widthb++) { //only 5 stated
                int sizeB = blockS[widthb]; //current block width
                dim3 block(sizeB, sizeB); //define block size
                dim3 grid((sizeN + sizeB - 1) / sizeB, (sizeN + sizeB - 1) / sizeB); //define grid size

                //declare CUDA events for timing
                float timef; //finalized part time kernel
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start); //recording the start time
                kernelMatrixMultiplicationGPU<<<grid, block >>>(deOP, deOM, deON, sizeN);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop); //synchronizing
                cudaEventElapsedTime(&timef, start, stop);

                //printing final text of everything
                printf("block: %d, gpu time: %.4f ms\n",
                    sizeB,
                    timef);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            free(hoM);
            free(hoN);
            free(hoP);
            cudaFree(deOM);
            cudaFree(deON);
            cudaFree(deOP);
        }
    }
    //exiting/return program
    return 0;
}

