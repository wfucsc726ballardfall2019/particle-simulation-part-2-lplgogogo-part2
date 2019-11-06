#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <vector>
#include <iostream>
#include "omp.h"
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
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
        {
            printf( "Options:\n" );
            printf( "-h to see this help\n" );
            printf( "-n <int> to set the number of particles\n" );
            printf( "-o <filename> to Sspecify the output file name\n" );
            printf( "-s <filename> to specify a summary file name\n" );
            printf( "-no turns off all correctness checks and particle output\n");
            return 0;
        }
    
    int n = read_int( argc, argv, "-n", 1000 );
    int thread = read_int(argc, argv, "-t", 1);
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    omp_set_num_threads(thread);
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );
    
    //calculate the gridSize, binSize, and then number of bin on one side;
    double gridSize = sqrt(n * density);
    double binSize = cutoff * 2;     // equals to the diameter of the circle
    int binNum = int(gridSize / binSize) + 1; // the binNum should be +1

    int NumberOfBins = binNum * binNum;

    vector<vector<int> >bin(NumberOfBins);

    //initialize 2d array for locks
    omp_lock_t binlocker[binNum][binNum];
    #pragma omp parallel for
    for (int i = 0; i < binNum; i++){
        for (int j = 0; j < binNum; j++){
            omp_init_lock(&binlocker[i][j]);
        }
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
	// #pragma omp parallel //private(dmin)
    // {
    for( int step = 0; step < NSTEPS; step++ )
    {
	    navg = 0;
        davg = 0.0;
	    dmin = 1.0;

        //put all the particles into corresponding bins
        #pragma omp parallel for 
        for (int i = 0; i < n; i++){
            int row = floor(particles[i].x / binSize);     //calculate the row index of the bin
            int col = floor(particles[i].y / binSize);     //calculate the column index of the bin
            //#pragma omp critical
            omp_set_lock(&binlocker[row][col]);
            bin[row * binNum + col].push_back(i);      //put the particle in to the bin in row major
            omp_unset_lock(&binlocker[row][col]);
        }

        //#pragma omp for reduction (+:navg) reduction(+:davg)
        #pragma omp parallel for
        for (int i = 0; i < n; i++){
            particles[i].ax = particles[i].ay = 0;      // initialize acceleration
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

        #pragma omp parallel for 
        for( int i = 0; i < n; i++ ){ 
            move( particles[i] );
        }	

        #pragma omp parallel for 
        for (int i = 0; i < binNum*binNum; i++){
            bin[i].resize(0);
        }	

        if( find_option( argc, argv, "-no" ) == -1 )
        {
        //
        // Computing statistical data
        //
            #pragma omp master
            if (navg) {
                absavg += davg/navg;
                nabsavg++;
            }
            #pragma omp critical
            if (dmin < absmin) absmin = dmin;
        
        //
        //  save if necessary
        //
            #pragma omp master
            if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, particles );
            }

    }
    
    #pragma omp parallel for
    for (int i = 0; i < binNum; i++){
        for (int j = 0; j < binNum; j++){
            omp_destroy_lock(&binlocker[i][j]);
        }
    }

    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
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
    if( fsum) 
        fprintf(fsum,"%d %d %g\n",n,thread,simulation_time);

 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );    
        free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
