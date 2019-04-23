/**
 * 
 * C S 4473 Homework 1
 * Fox's Algorithm
 * Daniel Schon
 * Robert Monaco
 * 
 **/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

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

//Chunk matrix into blocks based on p processes and n rows/columns
double** get_chunk(double** mat, int row, int col, int chunk_size){
  //Allocate memory for matrix based on chunk size 
  double **chunk = alloc_contiguous(chunk_size, chunk_size);

  //Iterate through chunk and assign value based on row and column offset
  for(int i = 0; i < chunk_size; i++){
    for(int j = 0; j < chunk_size; j++){
      chunk[i][j] = mat[i + row][j + col];
    }
  }

  return chunk;
}

// Multiply two matrices serially and return the result 
double** serial_multiply(double** matrix_a, double** matrix_b, int size)
{
  double** matrix_c = alloc_contiguous(size, size);
  for (int row = 0; row < size; row++)
  {
    for (int col = 0; col < size; col++)
    {
      int sum = 0;
      for (int i = 0; i < size; i++)
        sum += matrix_a[row][i]*matrix_b[i][col];
      matrix_c[row][col] = sum;
    }
  }
  return matrix_c;
}

// Add two matrices serially and return the result 
double** serial_add(double** matrix_a, double** matrix_b, int size)
{
  double** matrix_c = alloc_contiguous(size, size);
  for (int row = 0; row < size; row++)
  {
    for (int col = 0; col < size; col++)
    {
      matrix_c[row][col] = matrix_a[row][col] + matrix_b[row][col];
    }
  }
  return matrix_c;
}

// Method that computes the sum of 2 nxn matrices using Algorithm 2
int foxalgo(int n, int comm_sz, int my_rank, MPI_Comm comm)
{
  // Declare variables
  double **matrix_a,          // First multiplicand matrix
         **matrix_b,          // Second multiplicand matrix
         **matrix_c,          // Product matrix;
         **my_chunk_a,        // Local submatrix for each process
         **my_chunk_b,        // Local submatrix for each process
         **my_chunk_c,        // Local submatrix for solution matrix
         **recv_chunk;        // Local for result of submatrix mult
  double ***big_chungus;      // Buffer to store gathered chunks
  double start,               // Start time
         end;                 // End time
  int row,                    // row index 
      col,
      chunk_size,            // Number of doubles on a side of a chunk
      chunk_n;                // Number of chunks in a row or column

  // Seed the random number so each one is unique
  srand(time(NULL) * (my_rank + 1));

  // initialize arrays
  matrix_a = alloc_contiguous(n, n);
  matrix_b = alloc_contiguous(n, n);
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
      }
    }
  }

  // Wait for all processes to catch up and then start algorithm
  if (my_rank == 0)
    start = MPI_Wtime();

  chunk_size = n / sqrt(comm_sz);
  chunk_n = n / chunk_size;
  col = my_rank % chunk_n;
  row = (my_rank / chunk_n) % chunk_n;

  // Distribute chunks to process
  if (my_rank == 0)
  {
    // Grab a chunk for myself
    my_chunk_a = get_chunk(matrix_a, 0, 0, chunk_size);
    my_chunk_b = get_chunk(matrix_b, 0, 0, chunk_size);

    // Send chunks to all other processes
    for (int proc = 1; proc < comm_sz; proc++)
    {
      int chunk_col = (proc % chunk_n) * chunk_size;
      int chunk_row = ((proc / chunk_n) % chunk_n) * chunk_size;
      // Grab some chunks for the process
      get_chunk(matrix_a, chunk_row, chunk_col, chunk_size);
      get_chunk(matrix_b, chunk_row, chunk_col, chunk_size);

      // Send them with MPI
      MPI_Send(get_ptr(matrix_a),
        chunk_size*chunk_size,
        MPI_DOUBLE,
        proc,
        0,
        comm);
      MPI_Send(get_ptr(matrix_b),
        chunk_size*chunk_size,
        MPI_DOUBLE,
        proc,
        0,
        comm);
    }
  }
  else
  {
    // Receive my chunks from proc 0
    my_chunk_a = alloc_contiguous(chunk_size, chunk_size);
    my_chunk_b = alloc_contiguous(chunk_size, chunk_size);
    MPI_Recv(get_ptr(my_chunk_a), 
      chunk_size*chunk_size,
      MPI_DOUBLE,
      0, 
      0, 
      comm, 
      MPI_STATUS_IGNORE);
    MPI_Recv(get_ptr(my_chunk_b), 
      chunk_size*chunk_size,
      MPI_DOUBLE,
      0, 
      0, 
      comm, 
      MPI_STATUS_IGNORE);
  }

  // Allocate some local chunks for Fox's algorithm
  recv_chunk = alloc_contiguous(chunk_size, chunk_size);
  my_chunk_c = alloc_contiguous(chunk_size, chunk_size);

  //Iterate through n/chunk_size many steps for algorithm
  for (int step = 0; step < 1; step++)
  {
    for (int chunk_row = 0; chunk_row < chunk_n; chunk_row++)
    {
      // The column number of the process that will send in this row
      int sender_col = (chunk_row + step) % chunk_n;
      
      for(int chunk_col = 0; chunk_col < chunk_n; chunk_col++)
      {
        // Broadcast a chunk across the row
        if (col == sender_col)
        {
          //lööp through processes to determine which must send/recv
          for(int proc = 0; proc < comm_sz; proc++)
          {
            int proc_row = proc / chunk_n;
            if (proc_row == row)
            {
              // The process is in the row currently being broadcast
              if (proc != my_rank)
              {
                // Send chunk a to other procs in the row
                MPI_Send(get_ptr(my_chunk_a), 
                  chunk_size*chunk_size, 
                  MPI_DOUBLE, 
                  proc, 
                  0, 
                  comm);
              }
              else
              {
                // Copy my chunk a into my recv_chunk buffer
                recv_chunk = my_chunk_a;
              }
            }
          }
        }
        else
        {
          int sender_rank = row*chunk_n + sender_col;
          MPI_Recv(get_ptr(recv_chunk), 
            chunk_size*chunk_size,
            MPI_DOUBLE,
            sender_rank,
            0,
            comm,
            MPI_STATUS_IGNORE);
        }
      
        //Add the received chunk*B chunk value to the result matrix chunk C 
        my_chunk_c = serial_add(my_chunk_c, serial_multiply(recv_chunk, my_chunk_b, chunk_size), chunk_size);
      }
    }
  }
  
  //create a 3d buffer of chunks named big chungus
  if(my_rank == 0){
    big_chungus = malloc(comm_sz * sizeof(double**));
    for(int i = 0; i < comm_sz; i++){
      big_chungus[i] = alloc_contiguous(chunk_size, chunk_size);
    }
  }
  
  // Gather chunks into proc 0
  if (my_rank == 0)
  {
    // Copy my own chunk first
    big_chungus[0] = my_chunk_c;

    // Receive product chunks into chungus
    for (int p = 1; p < comm_sz; p++)
    {
      MPI_Recv(get_ptr(big_chungus[p]), 
        chunk_size*chunk_size,
        MPI_DOUBLE,
        p,
        0,
        comm,
        MPI_STATUS_IGNORE);
    }
  }
  else
  {
    // Send product chunk to proc 0
    MPI_Send(get_ptr(my_chunk_c), 
      chunk_size*chunk_size, 
      MPI_DOUBLE, 
      0, 
      0, 
      comm);
  }

  // Flatten Big Chungus into 2d matrix
  if (my_rank == 0)
  {
    int local_chunk;
    int local_row;
    int local_col;
    for (int global_row = 0; global_row < n; global_row++)
    {
      for (int global_col = 0; global_col < n; global_col++)
      {
        local_chunk = (global_row / chunk_size)*chunk_n + (global_col / chunk_size);
        local_row = global_row % chunk_size;
        local_col = global_col % chunk_size;
        matrix_c[global_row][global_col] = big_chungus[local_chunk][local_row][local_col];
      }
    }
  }

  // Algorithm is finished
  if (my_rank == 0)
  {
    end = MPI_Wtime();
    printf("%f\n", end - start);
  }

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

  if (my_rank == 0)
    printf("Processes: %d\n", comm_sz);

  for (int n = 100; n <= 1000; n += 100)
  {
    if (n % (int)(sqrt(comm_sz)) != 0)
    {
      if (my_rank == 0)
        printf("\n");
      continue;
    }
    foxalgo(n, comm_sz, my_rank, comm);
  }

  if (my_rank == 0)
    printf("\n");

  MPI_Finalize();
}