#include <stdio.h>
#include "mm-ymathModul.h"


const double EPSILON = 1e-4;

int sgn(double x){return (x>=0)?1:-1;}

double betrag(double x){return (x>0)?x:-x;}

double naiv_pow(double x, int n){
	double res = 1;
	for (int i = 0; i < n; i++)
		res*=x;
	return res;
}

double pow(double x, int n){return (n==0)?1:(n%2==1)?x*pow(x*x, (n-1)/2):pow(x*x, n/2);}

double pow_loop(double x, int n){
    double res = 1;
    while(n > 0)
        if (n%2==1){
            n = (n-1)/2;
            res*=x*x;
        }
        else{
            n/=2;
            res*=x;
        }
    return res;
}


long long fac(long n){
	long long res = 1;
	for (int i = 0; i < n; i++)
		res*=(i+1);
	return res;
}

double expo(double x){
    double res = 0;
    for(int k = 0; 1/(double)fac(k) > EPSILON; k++){
        res+=pow(x, k)/(double) fac(k);
    }
    return res;
}

double logarithm(double x){
    double res = 0;
    double temp = 1;
    for(int k = 0; temp > EPSILON; k++){
        temp=pow((x-1)/(x+1), 2*k+1)/(2*k+1);
        res+=temp;
        printf("%lf\n", temp);
    }
    return 2 * res;
}

double power(double x, double y){return expo(y*logarithm(x));}


double zeta_function(double s){
    double temp = 1;
    double res = 0;
    for(int k = 1; temp > EPSILON; k++){
        temp = 1.0/power(k, s);
        res+=temp;
    }
    return res;
}

double cos(double x){
	double res = 0;
	for(long k = 0; fac(2*k) > 0; k++)
		res+=((k%2==0)? 1:-1) *  power(x, 2*k)/(double)fac(2*k);
	return res;
}

double wurzel(double x){
	double X = x;
	while(X*X > x+EPSILON)
		X = 0.5*(X+x/X);
	return X;
}

double square_to(double *x){
    *x*=*x;
    return *x;
}

double root_to(double *x){
    *x = wurzel(*x);
    return *x;
}

int solveQuadraticEq(double *res1, double *res2, double a, double b, double c){
    double p = b/a;
    double q = c/a;
    double vel = p*p/4-q;
    if(vel < 0){
        return 1;
    }
    *res1 = -p/2+wurzel(vel);
    *res2 = -p/2-wurzel(vel);
    return 0;
}