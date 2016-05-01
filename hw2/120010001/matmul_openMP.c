//
//  matmul_openMP.c
//  
//
//  Created by Naveen Himthani on 02/03/16.
//
//

#include "matmul.h"
#include <omp.h>

#define CHUNK 15


/*** Function to initialise the matrices A and B to random floating points values ***/
void init_matrices(float** A, float** B, int i, int j, int chunk)
{
    int r1,r2;
    float r;
    srand((unsigned)time(NULL));
#pragma omp for schedule (dynamic, chunk)
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            r1 = rand();
            r2 = rand();
            r = (float)r1/r2;
            A[i][j] = r;
            r1 = rand();
            r2 = rand();
            r = (float)r1/r2;
            B[i][j] = r;
        }
    }
}

void init_pattern_matrices_omp(float **A, float** B, int i, int j, int chunk)
{
    
    #pragma omp for schedule (dynamic, chunk)
    for (i=0; i<N; i++){
        for (j=0; j<N; j++){
            A[i][j] = (float)(1);
            B[i][j] = (float)(1);
        }
    }
}

/*** Function to multiply 2 matrices A & B and store it in C ***/
void matrix_multiply(float** A, float** B, float** C, int i, int j, int k, int chunk, int tid)
{
#pragma omp for schedule (static, chunk)
    for (i=0; i<N; i++) {
        if (i%100==0)
        {
           // printf("approx %d Rows done\n", i);
        }
        for (j=0; j<N; j++) {
            C[i][j] = 0;
            for (k=0; k<N; k++) {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

int main()
{
    int tid, nthreads, i, j, k, chunk;
    chunk = CHUNK;
    
    struct timeval start, end;
    double time_spent;
    gettimeofday(&start, NULL);
    
    float**A = create_matrix();
    float**B = create_matrix();
    float**C = create_matrix();
    
#pragma omp parallel shared(A, B, C, nthreads, chunk) private(i,j,k,tid)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        printf("Number of Threads = %d\n", nthreads);
        init_pattern_matrices_omp(A,B,i,j,chunk);
        matrix_multiply(A,B,C,i,j,k,chunk,tid);
    }
    
    gettimeofday(&end, NULL);
    time_spent = ((end.tv_sec  - start.tv_sec) * 1000000u +
                  end.tv_usec - start.tv_usec) / 1.e6;
    
    printf("Time taken: %g\n",time_spent);
    printf("DONE\n");
    
    if (N<10) {
        print_all(A,B,C);
    }
    
    destroy_matrix(A);
    destroy_matrix(B);
    destroy_matrix(C);
    
    return 0;
}
