//Fox's algorithm serial implementation
//Bobby Monaco and Daniel Schon
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Generate a normally-distributed random number between -1.0 and 1.0
double random_normal()
{
  // Find the average of 12 uniformly random numbers. 
  // By Central Limit Theorem, this should approximate a normal distribution
  double sum = 0;
  int i;
  for (i = 0; i < 12; i++)
    sum += rand();
  sum /= 12;

  // Shift the range from [0.0, 1.0] to [-1.0, 1.0]
  sum = sum*2.0 - 1.0;

  return sum;
}

// Generate a matrix of size*size with random_normal values
int gen_mat(double * mat, int size){
    mat = malloc(size*size*sizeof(double));
    int i;
    
    for(i = 0; i<size*size;i++){
        mat[i] = random_normal();
    }
    
    return 0;
}

//Apply fox's algorithm to mat_a and mat_b and store result
//in mat_c
int fox(double * mat_a, double * mat_b, double * mat_c, int size){
    //processor rank from openMP
    int my_rank = omp_get_thread_num();
    //number of threads from openMP
    int thread_count = omp_get_num_threads();

    //size of grid block
    int block_size = (int)(size / sqrt((double)thread_count));
    
    int offset, i, a;

    #pragma omp parallel for
    for(a=0; a < size*size; a++){
        offset = (int)(a / size);
        i = (int)(a % size);

        mat_c[a] += mat_a[(size*(i + offset)) + i] * mat_b[size*offset + i];
    }
    
    return 0;
}

int main(int argc, char* argv[]){
    printf("here");
    //number of threads
    int thread_count = atoi(argv[1]);
    //size of matrix
    int n = atoi(argv[2]);
    
    //input matrices
    double * mat_a;
    double * mat_b;

    //iterator variable
    int iter;

    //matrix multiplication result
    double * result;

    //generate matrices
    gen_mat(mat_a,n);
    gen_mat(mat_b,n);
    printf("%d",0);

    result = malloc(n*n*sizeof(double));
    for(iter = 0; iter < n*n; iter++){
            result[iter] = 0.0;
    }

    printf("%d",1);

    # pragma omp parallel num_threads(thread_count)
    fox(mat_a, mat_b, result, n);

    printf("%d",2);

    for(iter = 0; iter < n*n; iter++){
        printf("%f",result[iter]);
        if(iter % n == 0){
            printf("\n");
        }
    }

    return 0;
}