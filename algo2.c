/**
 * 
 * C S 4473 Homework 1
 * Algorithm 2
 * Daniel Schon
 * Robert Monaco
 * 
 **/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Allocate a 2-dimensional array of doubles in contiguous memory
// This makes it much easier to send and receive with MPI
double** alloc_contiguous(int rows, int cols)
{
  // Allocate one contiguous memory block for data
  double* data = (double*)malloc(rows*cols*sizeof(double));
  // Allocate our matrix
  double** matrix = (double**)malloc(rows*sizeof(double*));
  // Point matrix cells to contiguous memory locations
  for (int i = 0; i < rows; i++)
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
  for (int i = 0; i < 12; i++)
    sum += rand();
  sum /= 12;

  // Shift the range from [0.0, 1.0] to [-1.0, 1.0]
  sum = sum*2.0 - 1.0;

  return sum;
}

// Method that computes the sum of 2 nxn matrices using Algorithm 2
int algorithm2(int n, int comm_sz, int my_rank, MPI_Comm comm)
{
  // Declare variables
  double **matrix_a,          // First multiplicand matrix
         **matrix_b,          // Second multiplicand matrix
         **matrix_b_col_maj,  // Column major order of matrix b
         **matrix_c,          // Product matrix;
         **local_rows,        // Local rows held by a single process
         **local_cols,        // Local columns help by a process
         **local_product;     // Product of the 2 above
  double start,               // Start time
         end;                 // End time
  int partition_sz,           // Number of numbers in a partition
      rows_per_partition;     // Number of rows (or columns) in a partition

  if (my_rank == 0)
    printf("Run: n=%d p=%d\n", n, comm_sz);

  // Seed the random number so each one is unique
  srand(time(NULL) * (my_rank + 1));

  // Initialize all the matrices
  matrix_a = alloc_contiguous(n, n);
  matrix_b = alloc_contiguous(n, n);
  matrix_b_col_maj = alloc_contiguous(n, n);
  matrix_c = alloc_contiguous(n, n);

  // Process 0 generates a random matrix
  if(my_rank == 0)
  {
    for (int y = 0; y < n; y++)
    {
      for (int x = 0; x < n; x++)
      {
        matrix_a[y][x] = random_normal();
        matrix_b[y][x] = random_normal();
        matrix_b_col_maj[x][y] = matrix_b[y][x];
      }
    }
  }

  // Run the algorithm serially and exit
  if (comm_sz == 1)
  {
    start = MPI_Wtime();
    for (int row = 0; row < n; row++)
    {
      for (int col = 0; col < n; col++)
      {
        int sum = 0;
        for (int i = 0; i < n; i++)
          sum += matrix_a[row][i]*matrix_b[col][i];
        matrix_c[row][col] = sum;
      }
    }
    end = MPI_Wtime();
    printf("Serial time: %f\n", end - start);
    return 0;
  }

  // Wait for all processes to catch up and then start algorithm 2
  MPI_Barrier(comm);
  if (my_rank == 0)
    start = MPI_Wtime();

  // Get partition sizes
  partition_sz = n*n / comm_sz;
  rows_per_partition = partition_sz / n;

  // Initialize local matrices
  local_rows = alloc_contiguous(rows_per_partition, n);
  local_cols = alloc_contiguous(rows_per_partition, n);
  local_product = alloc_contiguous(rows_per_partition, n);

  // Scatter rows and columns across all processes
  MPI_Scatter(get_ptr(matrix_a),
    partition_sz,
    MPI_DOUBLE,
    get_ptr(local_rows),
    partition_sz,
    MPI_DOUBLE,
    0,
    comm);
  MPI_Scatter(get_ptr(matrix_b_col_maj),
    partition_sz,
    MPI_DOUBLE,
    get_ptr(local_cols),
    partition_sz,
    MPI_DOUBLE,
    0,
    comm);

  // Algorithm 2 has p-1 steps
  for (int step = 0; step < comm_sz; step++)
  {
    int sum;
    // Get products of rows and columns
    for (int row = 0; row < rows_per_partition; row++)
    {
      sum = 0;
      for (int i = 0; i < n; i++)
      {
        sum += local_rows[row][i] * local_cols[row][i];
      }
      local_product[row][(my_rank + step) % n] = sum;
    }

    // Pass rows to the right
    MPI_Sendrecv_replace(get_ptr(local_rows), 
      partition_sz,
      MPI_DOUBLE, 
      (my_rank + 1) % comm_sz, 
      0, 
      (my_rank - 1 + comm_sz) % comm_sz, 
      0, 
      comm, 
      MPI_STATUS_IGNORE);
  }

  // Gather local products into final matrix
  MPI_Gather(get_ptr(local_rows),
    partition_sz,
    MPI_DOUBLE,
    get_ptr(matrix_c),
    partition_sz,
    MPI_DOUBLE,
    0,
    comm);

  // Algorithm is finished
  if (my_rank == 0)
  {
    end = MPI_Wtime();
    printf("%dx%d matrix computed in %f seconds\n", n, n, end - start);
  }

  // Deallocate memory
  free_contiguous(matrix_a);
  free_contiguous(matrix_b);
  free_contiguous(matrix_b_col_maj);
  free_contiguous(matrix_c);
  free_contiguous(local_rows);
  free_contiguous(local_cols);
  free_contiguous(local_product);
  return 0;
}

// The main method
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  // Declare and init MPI variables
  int comm_sz, my_rank;
  MPI_Comm comm;
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &comm_sz);
  MPI_Comm_rank(comm, &my_rank);

  for (int n = 100; n <= 1000; n += 100)
  {
    if (n % comm_sz != 0)
    {
      if (my_rank == 0)
        printf("matrix size %d not divisible by comm size %d. Skipping...\n", n, comm_sz);
      continue;
    }
    algorithm2(n, comm_sz, my_rank, comm);
  }

  MPI_Finalize();
  return 0;
}
