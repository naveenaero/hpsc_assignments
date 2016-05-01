//
//  matmul_mpi.c
//  
//
//  Created by Naveen Himthani on 10/03/16.
//
//

#include "matmul.h"
#include <mpi.h>



void init_matrices(float (*A)[N], float (*B)[N])
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


void init_pattern_matrices_mpi(float (*A)[N], float (*B)[N])
{
    int i,j;
    for (i=0; i<N; i++){
        for (j=0; j<N; j++){
            A[i][j] = (float)(1);
            B[i][j] = (float)(1);
        }
    }
}

void matrix_multiply(float (*A)[N], float (*B)[N], float (*C)[N], int start, int end, int proc_rank)
{
    int i,j,k;
    for (i=start; i<end; i++) {
       for (j=0; j<N; j++) {
            C[i][j] = 0;
            for (k=0; k<N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

}

void print_new_matrix(float (*M)[N])
{
    int i,j;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            printf("%f\t",M[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{   
    
    double time_taken, min_time, max_time, avg_time;
    int i,j,k;
    
    // Allocate memory to Matrices
    float (*A)[N] = malloc(sizeof(*A) * N);
    float (*B)[N] = malloc(sizeof(*B) * N);
    float (*C)[N] = malloc(sizeof(*C) * N);

    // Processor Number and Processor Rank
    int num_proc, proc_rank;
    // Tag for Send, Receive, Gather, Scatter
    int tag = 1;
    // MPI status
    MPI_Status status;
    // Number of Rows for each processor
    int proc_rows;

    // Initialise
    MPI_Init(&argc, &argv);
    
    // Set up the MPI proceses
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
//    if (proc_rank == 0)
//    {
//       printf("Process number %d says number of processes = %d\n", proc_rank, num_proc);
//    }
    
    /** Check if Matrix sizes exactly divides the number of processors alloted
     else calculate the number of Rows each Processor gets **/
    if (N%num_proc!=0) {
        if (proc_rank == 0) {
            printf("The matrix size does not divide into the number of processors exactly, exiting!");
        }
            MPI_Finalize();
            exit(-1);
        
    }
    else {
        proc_rows = N/num_proc;
    }
    
    
    /* If the processor rank is root (or zero) then initiate matrices with random numbers */
    if (proc_rank == 0) {
        init_pattern_matrices_mpi(A,B);
    }
    
    time_taken = MPI_Wtime();
    
    /* Assign start and end row numbers to each processor */
    int start, end;
    start = proc_rows*proc_rank;
    end = proc_rows*(proc_rank+1);
    
    MPI_Bcast(B, N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Scatter(A, proc_rows*N, MPI_FLOAT, A[start], proc_rows*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

//    printf("Process %d evaluating rows %d-%d\n", proc_rank, start, end-1);
    
    matrix_multiply(A, B, C, start, end, proc_rank);
    
    MPI_Gather(C[start], proc_rows*N, MPI_FLOAT, C, proc_rows*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

/*    if (proc_rank == 0 && N<=5) {
        printf("----------A---------\n");
        print_new_matrix(A);
        printf("----------B---------\n");
        print_new_matrix(B);
        printf("----------C---------\n");
        print_new_matrix(C);
    }
    else
	{ */
			//}
 
    time_taken = MPI_Wtime() - time_taken;
    MPI_Reduce(&time_taken, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_taken, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_taken, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   
	 if (proc_rank == 0){
	
	printf("C[first]=%f\nC[last]=%f\n",C[0][0], C[N-1][N-1]);
	avg_time /= num_proc;
        printf("\n Min Time = %g\n", min_time);
        printf("\n Max Time = %g\n", max_time);
        printf("\n Avg Time = %g\n", avg_time);
    }
    
    MPI_Finalize();

    free(A);
    free(B);
    free(C);
    
    return 0;
}
