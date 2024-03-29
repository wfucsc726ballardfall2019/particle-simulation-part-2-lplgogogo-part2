#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <vector>
#include <math.h>
#include <iostream>

using namespace std;

// constant copied from common.cpp for use
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin,rst;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-m", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    //cout << "number of procs is: " << n_proc << endl;
    


    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ )
        partition_offsets[i] = min( i * particle_per_proc, n );
    
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];
    
    //
    //  allocate storage for local partition
    //
    int nlocal = partition_sizes[rank];
    //particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
    
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    
    if( rank == 0 )
      init_particles( n, particles );
    //MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
    
    //broadcast particles to every thread
    // for (int i = 0; i < n; i++){
    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);
    //}
    //calculate the gridSize, binSize, and then number of bin on one side;
    double gridSize = sqrt(n * density);
    double binSize = cutoff * 2;     // equals to the diameter of the circle
    int binNum = int(gridSize / binSize) + 1; // the binNum should be +1
    int NumberOfBins = binNum * binNum;

    //initialize the grid
    vector<vector<int> > bin(NumberOfBins);
    //particle_t temp[n];

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    cout<<rank<<endl;
    for( int step = 0; step < NSTEPS; step++ )
    {
      

      navg = 0;
      dmin = 1.0;
      davg = 0.0;
      // 
      //  collect all global data locally (not good idea to do)
      //
      //MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
    
      //put all the particles into corresponding bins
      for (int i = 0; i < n; i++){
        int row = floor(particles[i].x / binSize);     //calculate the row index of the bin
        int col = floor(particles[i].y / binSize);     //calculate the column index of the bin
        bin[row * binNum + col].push_back(i);      //put the particle in to the bin in row major
      }

      //cout << "FLAG 1" << endl;
      //
      //  save current step if necessary (slightly different semantics than in other codes)
      //
      // if( find_option( argc, argv, "-no" ) == -1 )
      //   if( fsave && (step%SAVEFREQ) == 0 )
      //     save( fsave, n, particles );
      
      //
      //  compute all forces
      //
      //cout<<rank<<endl;

      for(int i = rank*particle_per_proc ; i<rank*particle_per_proc+nlocal ; i++){
        particles[i].ax = particles[i].ay = 0;
        int row = floor(particles[i].x / binSize);     //calculate the row index of the bin
        int col = floor(particles[i].y / binSize);     //calculate the column index of the bin
        for(int r = max(0,row -1); r<= min(row+1,binNum-1); r++){
          for(int c = max(0,col -1); c<= min(col+1,binNum-1); c++ ){
            for (int l = 0; l < bin[r*binNum + c].size(); l++){
              int fa = bin[r*binNum + c].at(l);
              apply_force(particles[i], particles[fa], &dmin, &davg, &navg);
            }
          }
        }
      }
    
      if( find_option( argc, argv, "-no" ) == -1 ){
        MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
        
        if (rank == 0){
          //
          // Computing statistical data
          //
          if (rnavg) {
            absavg +=  rdavg/rnavg;
            nabsavg++;
          }
          if (rdmin < absmin) absmin = rdmin;
        }
      }

      //
      //  move particles
      //

      for(int i = rank*particle_per_proc ; i<rank*particle_per_proc+nlocal ; i++){
        move(particles[i]);
      }

      // for (int i=0; i<n; i++){
      //   temp[i]();
      // }
      
      // for(int i = rank*particle_per_proc ; i<rank*particle_per_proc+nlocal ; i++){
      //   temp[i].x = particles[i].x;
      //   temp[i].y = particles[i].y;
      //   temp[i].ax = particles[i].ax;
      //   temp[i].ay = particles[i].ay;
      //   temp[i].vx = particles[i].vx;
      //   temp[i].vy = particles[i].vy;

      // }

      for (int i = 0; i < binNum*binNum; i++){
        bin[i].resize(0);
      }

      //cout << "FLAG 1" << endl;

      //MPI_Barrier(MPI_COMM_WORLD);

      for (int i = rank*particle_per_proc ; i<rank*particle_per_proc+nlocal ; i++){
        MPI_Gather(&particles[i], 1, PARTICLE, &particles[i], nlocal, PARTICLE, 0, MPI_COMM_WORLD);
        //MPI_Allgather(&temp[i], 1, PARTICLE, &temp[i], 1, PARTICLE, MPI_COMM_WORLD);
      }
      // MPI_Allgatherv(temp, n, PARTICLE, temp, n, partition_offsets, PARTICLE, MPI_COMM_WORLD);

      //for (int i = 0; i<n;i++){
      MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);
      //}
      
      // for(int i = 0; i<n; i++){
      //   particles[i].x = temp[i].x;
      //   particles[i].y = temp[i].y;
      //   particles[i].ax = temp[i].ax;
      //   particles[i].ay = temp[i].ay;
      //   particles[i].vx = temp[i].vx;
      //   particles[i].vy = temp[i].vy;
      // }
      
      //MPI_Barrier(MPI_COMM_WORLD);
      
      // if (rank != 0){
      //   free(particles);
      //   particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
      // }
      // for (int i = 0; i < n; i++){
      //   MPI_Bcast(&particles[i], 1, PARTICLE, 0, MPI_COMM_WORLD);
      // }
      //cout << "FLAG 2" << endl;

      if (rank == 0){
        if( find_option( argc, argv, "-no" ) == -1 ){
          if( fsave && (step%SAVEFREQ) == 0 ){
            save( fsave, n, particles );
          }
        }
      }
      //cout << "This is: " << step << "th iteration time." << endl;
    }
    simulation_time = read_timer( ) - simulation_time;
    MPI_Reduce(&simulation_time,&rst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 ){
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -The minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
        printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if(fsum){
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
      }
    }
  
    //
    //  release resources
    //
    if ( fsum )
      fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    //free( local );
    free( particles );
    if( fsave )
      fclose( fsave );
    MPI_Finalize( );
    return 0;
}
