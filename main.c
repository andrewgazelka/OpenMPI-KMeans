#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h> 

int read_csv_matrix(float *mtrx, char file_name[], int* nrow, int *nfeat);
void results_to_file(int *map2clust, int nsamples,int nfeat);
int MyKmeans_p(float *fdata, int *map2clust, int *counter, int *params,
		float tol, MPI_Comm comm);

#define NSAMPLES 10000
#define NFEAT 17
#define NCLUST 10    

int main(int argc, char *argv[]){
  int myid, nprocs;
  int Nc = 5; 
  float *fdata;
  int nloc, len, itmax=20;
  int i, j, ierr, nfeat = 0, nitems=0;
  char outfile[24], proc_name[40];
  FILE *fout;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  /*-------------------- file for reading data */
  //char file_name[] = "smaller.csv";
  char file_name[] = "/export/scratch/users/csci5451/pollution_Vsmall.csv";   
  int params[4];              // for passing a few parameters
  float tol = 1.e-10;         // stopping criterion
  int nmax = NSAMPLES;        // max number of samples to read
  int counter[NCLUST];        // counter for size of each cluster
  /*-------------------- START --------------------*/
  if (myid == 0) {
    fdata = (float *) malloc(nmax*NFEAT*sizeof(float));
    if (fdata == NULL)      exit(1);
    ierr=read_csv_matrix(fdata,file_name,&nitems, &nfeat);
    printf(" ierr %d\n",ierr);
    // get norm of fdata to define tol
    params[0] = nitems;
    params[1] = nfeat;
    MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD) ;
    nloc = (int) (nitems + nprocs-1)/nprocs;
    len = nloc*nfeat;
    MPI_Scatter(fdata,len,MPI_FLOAT,fdata,len,MPI_FLOAT,0,MPI_COMM_WORLD);
    printf(" myid %d , nloc %d, nitems %d, nfeat %d\n",myid,nloc,nitems,nfeat);
  }
  else{
    MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD) ; 
    nitems = params[0];
    nfeat = params[1]; 
    nloc = (int) (nitems + nprocs-1)/nprocs;
    fdata = (float *) malloc(nloc*nfeat*sizeof(float));
    len = nloc*nfeat;
    MPI_Scatter(fdata,len,MPI_FLOAT,fdata,len,MPI_FLOAT,0,MPI_COMM_WORLD);
    //    printf(" myid %d , nloc %d, nitems %d, nfeat %d\n",myid,nloc,nitems,nfeat);
  }
  //-------------------- adjust nloc for last rank */ 
  
  if (myid == nprocs-1)
    nloc = nitems - nloc*(nprocs-1);
  
  int map2clust[nloc];
    
  printf(" myid %d nsamples %d nloc %d  nfeat %d \n",myid,nitems,nloc,nfeat);

  params[0] = Nc; params[1]=nloc;  params[2]=nfeat;  params[3]=itmax;
  
  MPI_Barrier(MPI_COMM_WORLD);
    
  MyKmeans_p(fdata, map2clust, counter, params, tol,MPI_COMM_WORLD);

  sprintf(outfile, "OUT/FinalOutId%d",myid);
  printf(" myid %d my filename %s\n",myid,outfile);
  
  fout = fopen(outfile, "w" ) ;
  
  if (fout == NULL) {
    printf("Can't open output file %s...\n",outfile );
    exit(2);
  }
  /*-------------------- get processor name */
  MPI_Get_processor_name(proc_name,&i);  
  fprintf(fout,"proc name %s\n", proc_name);
  fprintf(fout,"Nc  %d,  nloc %d nfeat %d, Params \n",Nc,nloc,nfeat) ;
  if (myid == 0){
    fprintf(fout," =================cluster sizes \n");
    for (i=0; i<Nc; i++) 
      fprintf(fout,"Cluster %3d  size: %4d \n",i,counter[i]);
  }
  for (i=0; i<nloc; i++)
    fprintf(fout," %d \n",map2clust[i]);
  fprintf(fout," ---- \n");
  /*----------------------------- print out results clusters */
  for (i=0; i<nloc; i++){
    for (j=0; j<nfeat; j++)
      fprintf(fout," %9.6f ",fdata[i*nfeat+j]);
    fprintf(fout," \n ");
  }
  fclose(fout);
  MPI_Finalize();
  return(0); 
}


