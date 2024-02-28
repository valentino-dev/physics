#include <stdio.h>


double **matrix_alloc(int n){
	double **A = (double **)malloc(n * sizeof(double *));
	for(int i = 0; i < n; i++)
		A[i] = (double *)malloc(n * sizeof(double));
	return A;
}

void matrix_free(double **A, int n){
	for(int i = 0; i < n; i++)
		free(A[i]);
	free(A);
}

void matrix_print(double **A, int n){
	for(int i = 0; i < n-1; i++){
		for(int k = 0; k < n-1; k++)
			printf("%lf, ", A[i][k]);
		printf("%lf\n", A[i][n-1]);
	}
}

double **matrix_id(double **A, int n){
	double **A = matrix_alloc(n);
	for(int i = 0; i < n; i++)
		for(int k = 0; k < n; k++)
			if(k==i)
				A[i][k] = 1.0;
			else
				A[i][k] = 0.0;
	return A;
}


double **matrix_transpose(double **A, int n){
	double temp = 0;
	for(int i = 0; i < n; i++){
		for(int k = 0; k < i; k++){
			temp = A[i][k];
			A[i][k] = A[k][i];
			A[k][i] = temp;
		}
	}
	return A;
}

double **matrix_mult(double **A, double **B, int n){
	double **C = (double **)matrix_alloc(n);
	double temp_sum = 0;
	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++){
			temp_sum = 0;
			for(int k = 0; k < n; k++)
				temp_sum += A[i][k]*B[k][j];
			C[i][j] = temp_sum;
		}

	return C;
			
}

int main(){

}
