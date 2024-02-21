#include <stdio.h>

int isPrime(int c){
	for (int n = 2; n*n < c; n++)
		if (n%c==0)
			return 0;
	return 1;
}

int findNextPrime(int c){
	for(c++; !isPrime(c); c++);
	return c;
}

void Aufgabe2(){
	int c = 10;
	if (!isPrime(c))
		printf("Number %d is prime.\n", c);
	else
		printf("Number %d is not a prime.\n", c);
	printf("Next Prime is %d.\n", findNextPrime(c));
}


double findSqrt(double x){
	double epsilon = 0.0001;
	double X = x;
	while(X*X > x+epsilon)
		X = 0.5*(X+x/X);

	return X;
}

void Aufgabe3(){
	double x = 2;
	printf("The square root of %lf is %lf.\n", x, findSqrt(x));
}

int largestDevider(int a, int b){
	while(a!=0&&b!=0)
		if(a>b)
			a-=b;
		else
			b-=a;
	return (a==0)?b:a;
}

void Aufgabe4(){
	int a = 12;
	int b = 4;
	printf("Largest devider of a=%d and b=%d is %d.\n", a, b, largestDevider(a, b));

}

double pow(double x, int n){
	double res = 1;
	for (int i = 0; i < n-1; i++)
		res*=x;
	return res;
}

int fac(int n){
	int res = 1;
	for (int i = 0; i < n-1; i++)
		res*=(i+1);
	return res;
}

double cos(double x){
	int N = 5;
	double res = 0;
	for(int k = 0; k < N; k++)
		res+=(((k%2==0)? 1:-1) * pow(x, 2*k)/fac(2*k));
	return res;
}

void Aufgabe5(){
	double x = 10;
	printf("cos(%lf)=%lf", x, cos(x));
}

double Reihe(){
	double res = 0;
	double epsilon = 0.000001;
	for(int k = 1; res > 3.141*3.141/6-epsilon; res+=1/(k*k))
		continue;
	return res;
}

void Aufgabe6(){
	printf("The result of the series is %lf", Reihe());
}

int main(){
	Aufgabe3();
	return 0;
}
