//tristan guevarra - 20353857 - 21tag10
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main() { //initializing main function

	int countOfDevices; //initializing variable to store amount/count of cuda devices
    //using cudagetdevicecount() function
    cudaGetDeviceCount(&countOfDevices);
    printf("-------------------------ELEC374------------------------- \n");
    printf("cuda devices count: %d\n", countOfDevices); //printing devices count cuda
    //printing some nice add ons
    printf("--------------------------------------------------------- \n");
    //no for loop needed because we only search through one cuda device

    cudaDeviceProp properties1; //initializing a struct to obtain/hold properties of cuda device
    cudaGetDeviceProperties(&properties1, 0); //calling the properties of cuda 1 only

    printf("    number & type of cuda: 1 & %s \n", properties1.name); //printing out number and type of cuda device
    printf("    clock rate: %d \n", properties1.clockRate); //printing out clock rate
    printf("    number of streaming multiprocessors (SM): %d \n", properties1.multiProcessorCount); //printing out number of streaming multiprocessors
    printf("    number of cores: %d \n", properties1.multiProcessorCount * 64); //printing out number of cores
    printf("    warp size: %d \n", properties1.warpSize); //printing out warp size
    printf("    amount of global memory: %d \n", properties1.totalGlobalMem); //printing out amount of global memory
    printf("    amount of constant memory: %d \n", properties1.totalConstMem); //printing out amount of constant memory
    printf("    amount of shared memory per block: %d \n", properties1.sharedMemPerBlock); //printing out amount of shared memory per block
    printf("    number of registers available per block: %d \n", properties1.regsPerBlock); //printing out number of registers available per block
    printf("    maximum number of threads per block: %d \n", properties1.maxThreadsPerBlock); //printing out maximum number of threads per block
    printf("    maximum size of length dimension of a block: %d \n", properties1.maxThreadsDim[0]); //printing out maximum size of length dimension of a block
    printf("    maximum size of width dimension of a block: %d \n", properties1.maxThreadsDim[1]); //printing out maximum size of width dimension of a block
    printf("    maximum size of height dimension of a block: %d \n", properties1.maxThreadsDim[2]); //printing out maximum size of height dimension of a block
    printf("    maximum size of length dimension of a grid: %d \n", properties1.maxGridSize[0]); //printing out maximum size of length dimension of a grid
    printf("    maximum size of width dimension of a grid: %d \n", properties1.maxGridSize[1]); //printing out maximum size of width dimension of a grid
    printf("    maximum size of height dimension of a grid: %d \n", properties1.maxGridSize[2]); //printing out maximum size of height dimension of a grid
    //printing some nice add ons
    printf("---------------------------------------------------------- \n");


}