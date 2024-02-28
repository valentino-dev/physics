#include "arrayhelpers.h"
#include <stdio.h>

void printArray(int *arr, int size){
    printf("{");
    for(int i = 0; i < size-1; i++)
        printf("%d, ", arr[i]);
    printf("%d}\n", arr[size-1]);
}

int *initArr(int *arr, int size, int val){
    for(int i = 0; i < size; i++)
        *(arr+i) = val;
    return arr;
}

void copyAToB(int *A, int *B, int size){
    for(int i = 0; i < size; i++)
        B[i] = A[i];
}

int *roll(int *arr, int size, int shift){
    int temp[size];
    copyAToB(arr, temp, size);
    int *ptr;
    for(int i = 0; i < size; i++)
        arr[i] = temp[((i+shift)%size)];
    return arr;
}

int *reverse(int *arr, int size){
    int temp[size];
    copyAToB(arr, temp, size);
    for(int i = 0; i < size; i++){
        arr[i] = temp[size-i-1];
    }
    return arr;
}

int matchArr(int *arr, int arrSize, int *part, int partSize){
    for(int i = 0; i < arrSize; i++){
        int match = 1;
        for(int k = 0; k < partSize; k++){
            if(arr[(i+k)%arrSize]!=part[k]){
                match = 0;
                break;
            }
        }
        if(match)
            return i;
    }
    return -1;
}

int *bubbleSort(int *arr, int size){
    int sortet[size];
    initArr(sortet, size, 0);
    for(int i = 0; i < size; i++){
        int smol = -1;
        for(int k = 0; k < size; k++){
            if((arr[k] < smol || smol == -1) && (i == 0 | arr[k] > sortet[i-1])){
                smol = arr[k];
            }
        }
        sortet[i] = smol;
    }
    copyAToB(sortet, arr, size);
    return arr;

}

int *bucketSort(int *arr, int size, int bucket_size){
    int bucket[bucket_size];
    initArr(bucket, bucket_size, 0);
    for(int i = 0; i < size; i++)
        bucket[arr[i]]++;
    int c = 0;
    for(int i = 0; i < bucket_size; i++){
        for(int k = 0; k < bucket[i]; k++){
            arr[c+k] = i;
        }
        c += bucket[i];
    }
    return arr;
}


