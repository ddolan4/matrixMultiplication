#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <math.h>

/*
*   Author: Danielle Dolan worked with Anna Van Boven
*   version: 03.03.2021
*/

typedef struct threadArgs{
    int size;
    int range; 
    int col; 
    int row;
    double** matrix1;
    double** matrix2;
    double** result;
} threadArgs;

//Protypes
double** setMatrix(int size, double** matrix);
void printMatrix(int size, double** matrix);
void matrixMulti(int size, double** matrix1, double** matrix2, int numThreads);
double** regular(int size, double** matrix1, double** matrix2, double** result);
double* getCol(int size, int col, double** matrix);
double** makeMatrix(int size);
void freeMatrix(int sixe, double** matrix);
double** withThreads(int size, double** matrix1, double** matrix2, double** result, int numThreads);
void* multiplyRange(void* arg);
threadArgs* makeThreadArgs(int size, int col, int row, int range, double** matrix1, double** matrix2, double** result);
double checkForError(int size, double** result1, double** result2);


/*
*   A program to do matrix multiplcation. Takes two command line arguments: 1) the number of 
*   thread and 2) the size of the matrices. It will make two sqaure matrices of given size and 
*   fill them with random doubles from 0 to 99 inclusive. Will then profrom matrix multiplication
*   on them. It will do regular multiplcation 5 times and record the best time. If the numnber of
*   threads given is not 0, it will then do matrix multiplication using the numnber threads given,
*   running it 5 times and recording the best time. Will print the details out and then end
*
*/
int main(int argc, char *argv[]){ // int argc, char *argv[]
    if(argc < 3){ // check for too few command line arguments 
        printf("Sorry, too few agruments were supplied. Please try again and make sure that the two agurments are: \n");
        printf("1) The number threads to be used. Must be >= 0. \n");
        printf("2) The the size of matrices. Must be > 0");
        return 0;
    } else if( argc > 3){ // check for to many command line arguments
        printf("Sorry, to many agruments were supplied. Please try again and make sure that the two agurments are: \n");
        printf("1) The number threads to be used. Must be >= 0. \n");
        printf("2) The the size of matrices. Must be > 0");
        return 0;
    }
    int numThreads = atoi(argv[1]); // get number of threads
    int matrixSize = atoi(argv[2]); // get size of matrices 
    if(matrixSize <= 0){ // check that size is greater then 0
        printf("Sorry the size of the matrices cannot be less than or equal to 0. Please try again. \n");
        return 0;
    }
    // if no command line argument errors were hit, run as normal
    printf("Multiplying random matrices of size %dx%d\n", matrixSize, matrixSize);
    time_t t;
    srand((unsigned) time(&t)); // set srand with time to get randome values for matrix
    double** matrix1 = makeMatrix(matrixSize); // make matrix 1
    matrix1 = setMatrix(matrixSize, matrix1); // fill matrix 1 with random numbers 
    double** matrix2 = makeMatrix(matrixSize); // make matrix 2
    matrix2 = setMatrix(matrixSize, matrix2); // fill matrix 2 with random numbers
    matrixMulti(matrixSize, matrix1, matrix2, numThreads); // call to do matrix multiplication
    freeMatrix(matrixSize, matrix1); // free matrix 1
    freeMatrix(matrixSize, matrix2); // free matrix 2
    return 0;
}

/*
*   Will make a square 2D array of doubles to represnt a matrix. Expects an int for the size
*
*/
double** makeMatrix(int size){
    double** matrix = (double **)malloc(size * sizeof(double *));
    for(int i=0; i<size; i++){
        matrix[i] = (double *)malloc(size *sizeof(double));
    }
    return matrix;
}

/*
*   Will fill a given matrix with random doubles between 0 and 99 inclusive. 
*   Expects an int for the size and a matrix of doubles
*
*/
double** setMatrix(int size, double** matrix){
    double range = 100; // range is 0 to 99 inclusive
    for(int i=0; i<size; i++){ // for each row
        for(int j=0; j<size; j++){ // for each col
            // get new random double based on time
            double random = (rand() / (double) RAND_MAX)  * range; 
            matrix[i][j] = random; // set matrix at (i,j) = to random value
        }
    }
    return matrix;
}

/*
*   Preforms matrix multiplication on two matrices of doubles. Will do regular matrix multiplication without
*   threads 5 times and record the best time. If the number of threads given is greater then 0 it will also 
*   preform matrix multiplication using that number of threads 5 times and record the best time. It will then 
*   compare the two for speed up factor and errors. Expects an int for the size, two matrices of doubles, and
*   an int for the number of threads.
*
*/
void matrixMulti(int size, double** matrix1, double** matrix2, int numThreads){
    long bestNotThreading = 10000000000; // set obscure value for original
    long bestWithThreading = 10000000000; // obscure value for original
    struct timeval start, end; // set up times

    // do regular multiplication with no threads 5 times to get best value
    double** result1 = makeMatrix(size);
    for(int i=0; i<5; i++){ 
        gettimeofday(&start, NULL); // gete start time
        result1 = regular(size, matrix1, matrix2, result1); // preform without threads
        gettimeofday(&end, NULL); // get end time
        long seconds = (end.tv_sec - start.tv_sec); // get time elapsed in seconds
        long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec); // get time in mircoseconds
        if(micros < bestNotThreading){ // check for new best time without threads
            bestNotThreading = micros;
        }
    }
    printf("Best time without threading: %ld microseconds.\n", bestNotThreading); 

    // if there are not 0 threads also do matrix multiplication with number of threads given
    if(numThreads != 0){
        double** result2 = makeMatrix(size); // make result matrix
        for(int i=0; i<5; i++){
            gettimeofday(&start, NULL); // gete start time
            result2 = withThreads(size, matrix1, matrix2, result2, numThreads); // prefrom with threads
            gettimeofday(&end, NULL); // get end time
            long seconds = (end.tv_sec - start.tv_sec); // get time elapsed in seconds
            long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec); // get time in mircoseconds
            if(micros < bestWithThreading){ // check for new best time with threads
                bestWithThreading = micros;
            }
        }
        // report on time, speed up, and error
        printf("Best time with %d threads: %ld microseconds.\n", numThreads, bestWithThreading);
        double speedUp = (double) bestNotThreading/bestWithThreading;
        printf("Observed speedup is a factor of %f microseconds.\n", speedUp);
        double check = checkForError(size, result1, result2);
        printf("Observed error is %f\n", check);
        freeMatrix(size, result2); // free result of math with threads 
    }
    freeMatrix(size, result1); // free result of math without threads
}

/*
*   Makes a new threadArgs struct and returns it. Expects an int for the size, and int for the range, and int
*   for the col, an int for the row, a matrix1 of doubles, a matrix2 of doubles and a result matrix of doubles
*
*/
threadArgs* makeThreadArgs(int size, int col, int row, int range, double** matrix1, double** matrix2, double** result){
    threadArgs *current = (threadArgs*)malloc(sizeof(threadArgs)); // make room on heap
    // set all values
    current->size = size;
    current->col = col;
    current->row = row;
    current->range = range;
    current->matrix1 = matrix1;
    current->matrix2 = matrix2;
    current->result = result;
    return current;
}

/*
*   Returns a matrix that is the result of using threads to compute the matrix multiplication of two matrices. 
*   expects an int for the size, and three double matrices- two bring the ones that math will be preformed on, and 
*   one to hold the result - and an int for the number of threads
*
*/
double** withThreads(int size, double** matrix1, double** matrix2, double** result, int numThreads){
    // set up an structures to hold threads
    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * numThreads); //
    threadArgs* threadArgsStructs[numThreads]; 

    // calculate work to be done by each thread
    int dividedWork = (size * size) / numThreads;
    int extra = (size * size) % numThreads; // check for left overs 
    // set inital col and row
    int row = 0;
    int col = 0;
    for(int i=0; i<numThreads; i++){ // for each thread
        int range = dividedWork; // set range of work to be done
        if(extra != 0){ // add extra if needed
            range ++;
            extra --;
        }
        // create theard args so it has access to everything
        threadArgsStructs[i] = makeThreadArgs(size, col, row, range, matrix1, matrix2, result);
        // spin off threads 
        pthread_create(&threads[i], NULL, multiplyRange, (void *)threadArgsStructs[i]);
        // change row and col by range
        row += (range+col)/size; 
        col = (range+col)%size;
    }   
    // rejoin them, free data
    for (int i=0; i<numThreads; i++) {
        pthread_join(threads[i], NULL);
        free(threadArgsStructs[i]);
    }
    free(threads); 
    return result;
}

/*
*   Call back function for threads to compute the dot product for a given range of a matrix. Expects a threadArd
*   so that all infromation can be accessed. 
*
*/
void* multiplyRange(void* arg){
    threadArgs *current = arg; // set as threadArg
    // get needed values
    int size = current->size; 
    int range = current->range;
    int row = current->row;
    int col = current->col;
    while(range > 0){ // while there are still spots to fill
        double* currentRow = current->matrix1[row]; // get row of matrix 1
        double* column = getCol(size, col, current->matrix2); // get col of matrixs
        double newValue = 0;
        // compute dot product
        for(int i=0; i<size; i++){
            newValue += currentRow[i] * column[i];
        }
        current->result[row][col] = newValue; // set result at (i, j) to dot product
        col++; // change col
        if(col == size){ // check for end of matrix
            row++;
            col = 0;
        }
        range--; // change range
    }
    pthread_exit(0); // end of threads job
}

/*
*   Returns the result of multiplying two matrices. Expects an int for the size of the matrices and
*   two matrices of doubles of that size.
*
*/
double** regular(int size, double** matrix1, double** matrix2, double** result){
    for(int i=0; i<size; i++){ // for each row
        double *row = matrix1[i]; // get current row
        for(int j=0; j<size; j++){ // for each col
            double* col = getCol(size, j, matrix2); // get current col
            double newValue = 0;
            for(int h=0; h<size; h++){ // get dot product of row and col
                newValue += (row[h] * col[h]);
            }
            result[i][j] = newValue; // set reslt at (i, j) = to the dotproduct 
            free(col);
        }
    }
    return result;
}

/*
*   Returns the sum of squares between the two matrices. If the two matrices are the same the 
*   retunred value should equal 0. Expects an int for the size and two matrices of doubles
*
*/
double checkForError(int size, double** result1, double** result2){
    double check = 0;
    for(int i=0; i<size; i++){ // for each row
        for(int j=0; j<size; j++){ // for each col
            check += pow((result1[i][j] - result2[i][j]), 2); // sum of sqaure for each position
        }
    }
    return check;
}

/*
*   Will free a matrix from the heap. Expects an int for the size of the matrix and a matrix
*   of doubles
*
*/
void freeMatrix(int size, double** matrix){
    for(int i=0; i<size; i++){ // free each column
        free(matrix[i]);
    }
    free(matrix); // free the whole thing;
}

/*
*   Returns an array of doubles represnting the column of a matrix. Expects an int for the size, 
*   and int for the colum to get, and a matrix of doubles
*
*/
double* getCol(int size, int col, double** matrix){
    double* cols = (double*)malloc(size * sizeof(double)); // make room for column
    for(int i=0; i<size; i++){ // loop through rows to get the value at each column
        cols[i] = matrix[i][col]; // add to array
    }
    return cols; 
}

/*
*   prints out a matrix in its square formation. Expects and int for the matrix size and 
*   a matrix of doubles of that size
*   
*/
void printMatrix(int size, double** matrix){
    for(int i=0; i<size; i++){ // rows
        for(int j=0; j<size; j++){ // cols
            printf("%f ", matrix[i][j]); // print out value at (rol, col);
        }
        printf("\n");
    }
}