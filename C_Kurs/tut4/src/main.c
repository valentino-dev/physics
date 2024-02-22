#include <stdio.h>
#include "mm-ymathModul.h"
#include "arrayhelpers.h"

void aufgabe4(){
	const int size = 10;
	int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	printArray(arr, size);
	int arr2[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	printArray(initArr(arr2, size, 4), size);
	int arr3[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	printArray(roll(arr3, size, 4), size);
	int arr4[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	printArray(reverse(arr4, size), size);

	int A[10] = {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10};
	int B[3] = {4, 5, 6};
	int C[2] = {5, 7};
	int D[2] = {9, 10};
	printf("stelle %d\n", matchArr(A, 10, B, 3));
	printf("stelle %d\n", matchArr(A, 10, C, 2));
	printf("stelle %d\n", matchArr(A, 10, D, 2));
}

void aufgabe5(){
	const int size = 10;
	int arr[10] = {4, 2, 7, 1, 9, 3, 5, 6, 10, 8};
	printf("Bubble Sort: \n");
	printArray(bubbleSort(arr, size), size);
	int arr2[10] = {4, 2, 7, 7, 9, 3, 5, 6, 10, 8};
	printf("Bucket Sort: \n");
	printArray(bucketSort(arr2, size, 20000), size);
 
}

int main(){
	printf("Aufgabe 4:\n");
	aufgabe4();
	printf("Aufgabe 5:\n");
	aufgabe5();
	return 0;
}
