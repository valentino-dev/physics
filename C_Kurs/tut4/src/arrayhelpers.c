#include "arrayhelpers.h"
#include <stdio.h>

void printArray(int *arr, int size){
    for(int i = 0; i < size; i++)
        printf("%d\n", *(arr+i));
}

int *initArr(int *arr, int size, int val){
    for(int i = 0; i < size; i++)
        *(arr+i) = val;
    return arr;
}

void copyArr(int *A, int *B, int size){
    for(int i = 0; i < size; i++)
        B[i] = A[i];
}

int *roll(int *arr, int size, int shift){
    int temp[size];
    copyArr(arr, temp, size);
    int *ptr;
    for(int i = 0; i < size; i++)
        arr[i] = temp[((i+shift)%size)];
    return arr;
}

int *reverse(int *arr, int size){
    //printf("size %d\n", size);
    int temp[size];
    copyArr(arr, temp, size);
    for(int i = 0; i < size; i++){
        //int idx = ((-i)%size);
        //printf("%d\n", idx);
        arr[i] = temp[size-i-1];
        printf("%d\n", arr[i]);
    }
    return arr;
}

int matchArr(int *arr, int arrSize, int *part, int partSize){
    int match = 1;
    for(int i = 0; i < arrSize; i++){
        match = 0;
        for(int k = 0; k < partSize; k++){
            if(arr[(i+k)%arrSize]!=part[k])
                match = 0;
        }
        if(match)
            return i;
    }
    return -1;
}


