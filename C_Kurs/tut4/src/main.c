#include <stdio.h>
#include "mm-ymathModul.h"
#include "arrayhelpers.h"

int main(){
	double x = 2.0;
	square_to(&x);
	//printf("Sq: %lf\n", x);
	root_to(&x);
	root_to(&x);
	//printf("Sqrt: %lf\n", x);
	double res1;
	double res2;
	solveQuadraticEq(&res1, &res2, 1.0, 4.0, 3.0);
	//printf("res1 %lf; res2 %lf\n", res1, res2);

	const int size = 10;
	int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

	//printArray(arr, size);
	//printArray(initArr(arr, size, 4), size);
	//printArray(roll(arr, size, 4), size);
	printArray(reverse(arr, size));

}
