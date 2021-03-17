#pragma once

int MyKmeans_p(float *fdata, int *map2clust, int *counter, const int *params,
               float tolerance, MPI_Comm comm);


int saxpy_(int *n, float *a, float *x, int *incx, float *y, int *incy);

int sscal_(int *n, float *alpha, float *x, int *inc);

void scopy_(int *nfeat, float *x, int *incx, float *y, int *incy);

void get_rand_ftr(float *ctr, float *fdata, int sampleCount, int featureCount);
