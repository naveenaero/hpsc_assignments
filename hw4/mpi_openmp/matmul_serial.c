//
//  matmul_serial.c
//
//
//  Created by Naveen Himthani on 02/03/16.
//
//

#include "matmul.h"


void init_matrices(float** A, float** B)
{
    int i,j;
    int r1,r2;
    float r;
    srand((unsigned)time(NULL));
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


void matrix_multiply_serial(float** A, float** B, float** C)
{
    int i,j,k;
    for (i=0; i<N; i++) {
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
    struct timeval start, end;
    double time_spent;
    gettimeofday(&start, NULL);
    
    float**A = create_matrix();
    float**B = create_matrix();
    float**C = create_matrix();
    
    init_pattern_matrices_serial(A,B);
    matrix_multiply_serial(A,B,C);
    
    printf("DONE\n");
    gettimeofday(&end, NULL);
    time_spent = ((end.tv_sec  - start.tv_sec) * 1000000u +
             end.tv_usec - start.tv_usec) / 1.e6;
    
    printf("Time taken: %g\n",time_spent);
    if (N<5){
        print_all(A,B,C);
    }
    return 0;
}
