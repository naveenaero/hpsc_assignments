//
//  matmul_openMP.h
//  
//
//  Created by Naveen Himthani on 02/03/16.
//
//

#ifndef matmul_openMP_h
#define matmul_openMP_h

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 1000

/*** Function to allocate memory to create a Matrix ***/
float** create_matrix()
{
    float** M=(float**)malloc(N*sizeof(float *));
    int i;
    for(i=0; i<N;i++)
    {
        M[i]=(float*)malloc(N*sizeof(float));
    }
    return M;
}

/*** Function to destroy a Matrix ***/
void destroy_matrix(float** M)
{
    for (int i=0; i<N; i++) {
        float* M_row = M[i];
        free(M_row);
        
    }
}

void init_pattern_matrices_serial(float **A, float** B)
{
    int i,j;
    for (i=0; i<N; i++){
        for (j=0; j<N; j++){
            A[i][j] = (float)(1);
            B[i][j] = (float)(1);
        }
    }
}

/*** Function to print a Matrix ***/
void print_matrix(float** M)
{
    int i,j;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            printf("%f\t",M[i][j]);
        }
        printf("\n");
    }
}

/*** Function to print all Matrices ***/
void print_all(float** A, float** B, float** C)
{
    printf("-------Matrix A--------\n"); print_matrix(A);
    printf("-------Matrix B--------\n"); print_matrix(B);
    printf("-------Matrix C--------\n"); print_matrix(C);
    
}


#endif /* matmul_openMP_h */
