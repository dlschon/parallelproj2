/**
 * 
 * C S 4473 Project 2
 * Algorithm 2
 * Daniel Schon
 * Robert Monaco
 * 
 **/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Allocate a 2-dimensional array of doubles in contiguous memory
double** alloc_contiguous(int rows, int cols)
{
  // Allocate one contiguous memory block for data
  double* data = (double*)malloc(rows*cols*sizeof(double));
  // Allocate our matrix
  double** matrix = (double**)malloc(rows*sizeof(double*));
  // Point matrix cells to contiguous memory locations
  int i;
  for (i = 0; i < rows; i++)
    matrix[i] = &(data[cols*i]);

  return matrix;
}

// Get a pointer to a matrix so we can send or receive it
double* get_ptr(double** matrix)
{
  return &(matrix[0][0]);
}

// Free memory used by a contiguous matrix
void free_contiguous(double** matrix)
{
  // Free the data array and the nested array itself
  free(matrix[0]);
  free(matrix);
}


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

// Method that computes the sum of 2 nxn matrices using Algorithm 2
int algorithm2(int n, int thread_count)
{
  // Declare variables
  double **matrix_a,          // First multiplicand matrix
         **matrix_b,          // Second multiplicand matrix
         **matrix_c;          // Product matrix;
  double start,               // Start time
         end;                 // End time

  // Seed the random number so each one is unique
  srand(time(NULL));

  // Initialize all the matrices
  matrix_a = alloc_contiguous(n, n);
  matrix_b = alloc_contiguous(n, n);
  matrix_c = alloc_contiguous(n, n);

  // generate a random matrix
  int x,y;
  for (y = 0; y < n; y++)
  {
    for (x = 0; x < n; x++)
    {
      matrix_a[y][x] = random_normal();
      matrix_b[y][x] = random_normal();
    }
  }

  // Run the algorithm 
  start = omp_get_wtime();
  int rowcol;
  int row;
  int col;
  # pragma omp parallel for num_threads(thread_count)
  for (rowcol = 0; rowcol < n*n; rowcol++)
  {
    col = rowcol % n;
    row = rowcol / n;

    int sum = 0;
    int i;
    for (i = 0; i < n; i++)
      sum += matrix_a[row][i]*matrix_b[col][i];
    matrix_c[row][col] = sum;
  }
  end = omp_get_wtime();
  printf("time: %f\n", end - start);
  return 0;
}

int main(int argc, char** argv)
{
  int thread_count, n;
  thread_count = atoi(argv[1]);

  printf("p = %d\n", thread_count);

  for (n = 100; n <= 1000; n += 100)
    algorithm2(n, thread_count);

  return 0;
}