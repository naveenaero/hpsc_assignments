#include<stdio.h>
#include<omp.h>
#include<math.h>


int main()
{
int i;
double x_0 = 0.5;
double h = 0.01;
double sum=0;
int N = 10;

#pragma omp  printf("Hello\n");

//#pragma omp parallel
{
#pragma omp parallel for
for(i=0; i<N; ++i)
{
	double x = x_0 + i*h + h/2;
#pragma omp critical
{
	sum += sqrt(1-x*x);	
}
}
}
printf("Sum:%g\n",sum);
return 0;
}
