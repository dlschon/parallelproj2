///Fox's algorithm serial implementation
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
    int i;
    
    for(i = 0; i<size*size;i++){
        mat[i] = random_normal();
    }
    
    return 0;
}

//Apply fox's algorithm to mat_a and mat_b and store result
//in mat_c
int fox(double * mat_a, double * mat_b, double * mat_c, int size, int thread_count){

    int offset, i, a;
    #pragma omp parallel for num_threads(thread_count) private(offset, i)
    for(a=0; a < size*size; a++){
        offset = (int)(a / size);
        i = (int)(a % size);
        mat_c[a] += mat_a[(size*i) + ((i + offset) % size)] * mat_b[size*((i + offset) % size) + i];
    }
    
    return 0;
}

int main(int argc, char** argv){
    //number of threads
    int thread_count = atoi(argv[1]);

    double start_time, elapsed_time;

    //iterator variable
    int iter;

    //matrix multiplication result
    double * result;

    int n;
    printf("%d\n",thread_count);
    for(n=100;n<1100;n+=100){
      if((n % (int)sqrt(thread_count) == 0)){
        //input matrices
        double * mat_a = malloc(n*n*sizeof(double));
        double * mat_b = malloc(n*n*sizeof(double));
        //generate matrices
        gen_mat(mat_a,n);
        gen_mat(mat_b,n);

        result = malloc(n*n*sizeof(double));
        for(iter = 0; iter < n*n; iter++){
                result[iter] = 0.0;
        }
        start_time = omp_get_wtime();
        
        fox(mat_a, mat_b, result, n, thread_count);
        //calculate runtime from wall clock
        elapsed_time = omp_get_wtime() - start_time;
        printf("%d:%f\n", n, elapsed_time);
      }
    }

    return 0;
}