#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initGame(int *Z, int size){
	srand(42);
	for(int i = 0; i < size; i++){
		Z[i] = rand()%2;
	}
}

int stepCell(int *Z, int i0, int k0, int d1, int d2){
	int alive_cells = 0;
	for(int i = i0-1; i < i0+2; i++){
		for(int k = k0-1; k < k0+2; k++){
			if(Z[(i+d1)%d1 * d2 + (k+d2)%d2])
				alive_cells++;
		}
	}
	if(Z[i0, k0])
		return (alive_cells == 2 || alive_cells == 3)?1:0;
	
	return (alive_cells == 3)?1:0;
}

int *step(int *Z0, int *Z1, int d1, int d2){
	for(int i = 0; i < d1; i++){
		for(int k = 0; k < d2; k++){
			Z1[i * d2 + k] = stepCell(Z0, i, k, d1, d2);
		}
	}
	return Z1;
}

void printZ(int *Z, int d1, int d2){
	for(int i = 0; i < d1; i++){
		for(int k = 0; k < d2; k++){
			printf("%d", Z[i*d2+k]);
		}
		printf("\n");
	}
}

int sum(int *Z, int size){
	int sum = 0;
	for(int i = 0; i < size; i++){
		sum+=Z[i];
	}
	return sum;
}

int game(){
	const int d1 = 20;
	const int d2 = 20;
	int size = d1*d2;

	int Z0[d1*d2], Z1[d1*d2];
	int *Ztau = Z0;
	int *Ztaup1 = Z1;
	initGame(Z0, size);

	system("clear");
	printZ(Ztau, d1, d2);

	while(sum(Z0, size) != 0){
		usleep(1000000);
		system("clear");
		step(Ztau, Ztaup1, d1, d2);
		printZ(Ztaup1, d1, d2);
		int *temp = Ztau;
		Ztau = Ztaup1;
		Ztaup1 = temp;
	}

}


int main(){
	game();
	return 0;
}
