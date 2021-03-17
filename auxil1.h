#pragma once

int MyKmeans_p(float *fdata, int *map2clust, int *counter, const int *params,
               float tolerance, MPI_Comm comm);

