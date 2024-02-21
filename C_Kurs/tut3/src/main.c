#include <stdio.h>
#include "omod.h"
void Aufgabe1(){
	double x = 3.141*3/2;
	x = 2;
	printf("%lf %lf\n", pow(1, 3), (double)fac(3));
	//printf("x=%lf => sgn(x)=%d, betragf(x)=%lf, cos(x)=%lf, wurzel(|x|)=%lf.\n", x, sgn(x), betrag(x), cos(x), wurzel(betrag(x)));
	printf("expo: %lf", zeta_function(2.0));
}

int main(){
	Aufgabe1();
	return 0;
}
