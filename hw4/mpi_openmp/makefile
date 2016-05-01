MPI_SRC = matmul_mpi.c
OMP_SRC = matmul_openMP.c
OMP_OBJ = matmul_omp
MPI_OBJ = matmul_mpi
OPENMP=-fopenmp
MPICC=mpicc
MPIRUN=mpirun
NP=4
N=100
MPI_OUT = mpi_out.txt
OMP_OUT = omp_out.txt
CC=gcc
VALGRIND = valgrind
VALFLAG = --tool=callgrind --simulate-cache=yes
DEBUG=-pg

all : $(MPI_OBJ) $(OMP_OBJ)
	time $(VALGRIND) $(VALFLAG) $(MPIRUN) -np $(NP) $(MPI_OBJ) > $(MPI_OUT)
	time $(VALGRIND) $(VALFLAG) ./$(OMP_OBJ) > $(OMP_OUT) 

$(MPI_OBJ) : $(MPI_SRC)
	$(MPICC) $(DEBUG) -DN=$(N) -o $(MPI_OBJ) $(MPI_SRC) -std=c99

$(OMP_OBJ) :  $(OMP_SRC)
	$(CC) $(DEBUG) -DN=$(N) -fopenmp -o $(OMP_OBJ) $(OMP_SRC) -std=c99

clean:
	rm -r $(MPI_OBJ) $(OMP_OBJ)

