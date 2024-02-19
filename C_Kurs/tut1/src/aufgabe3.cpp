#include <stdio.h>

int main(){
	int n = 10;
	if(n%2==0){
		printf("n=%d ist gerade => %lf\n", n, (double) n/2);
	}
	else
	{
		printf("n=%d ist ungerade => %lf\n", n, (double) (n+1)/2);
	}

	return 0;
}
