#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "float.h"
#include "assert.h"

#define MAX_LINE 210
#define bool char
#define true 1
#define false 0
#define assert__(x) for ( ; !(x) ; assert(x) )

int MyKmeans_p(const float *fdata, int *map2clust, int *counter, const int *params,
               float tolerance, MPI_Comm comm);


int saxpy_(int *n, float *a, float *x, int *incx, float *y, int *incy);

int sscal_(int *n, float *alpha, float *x, int *inc);

void scopy_(int *nfeat, float *x, int *incx, float *y, int *incy);

void get_rand_ftr(float *ctr, float *fdata, int sampleCount, int featureCount);

void get_rand_ftr(float *ctr, float *fdata, int sampleCount, int featureCount) {
// gets a random convex  combination of all samples
    float tot, t;
    int j, one = 1;
    /*-------------------- initialize to zero */
    for (j = 0; j < featureCount; j++)
        ctr[j] = 0.0;
    tot = 0.0;
    /*-------------------- loop over all samples*/
    for (j = 0; j < sampleCount; j++) {
        t = (float) (rand() / (float) RAND_MAX);
        t = t * t;
        //if (t < 0.5){
        //    for (k=0; k<featureCount; k++)      ctr[k] += t*fdata[j*featureCount+k];
        saxpy_(&featureCount, &t, &fdata[j * featureCount], &one, ctr, &one);
        tot += t;
    }
    tot = 1.0 / tot;
    sscal_(&featureCount, &tot, ctr, &one);
}


int assign_ctrs(float *dist, int k) {
    float min;
    min = dist[0];
    int i, ctr = 0;
    for (i = 1; i < k; i++) {
        if (min > dist[i]) {
            ctr = i;
            min = dist[i];
        }
    }
    return ctr;
}

/*-------------------- reading data */
int read_csv_matrix(float *mtrx, char file_name[], int *nrow, int *nfeat) {
/* -------------------- reads data from a csv file to mtrx */
    FILE *finputs;
    char line[MAX_LINE], subline[MAX_LINE];

    if (NULL == (finputs = fopen(file_name, "r")))
        exit(1);
    memset(line, 0, MAX_LINE);
    //
    int k, j, start, rlen, lrow = 0, lfeat = 0, jcol = 0, jcol0 = 0, len, first = 1;
    const char *delim;
    delim = ",";
    /*-------------------- big while loop */
    while (fgets(line, MAX_LINE, finputs)) {
        if (first) {
//--------------------ignore first line of csv file 
            first = 0;
            continue;
        }
        len = strlen(line);
        lrow++;
        start = 0;
/*-------------------- go through the line */
        for (j = 0; j < len; j++) {
            if (line[j] == *delim || j == len - 1) {
                k = j - start;
                memcpy(subline, &line[start], k * sizeof(char));
                //-------------------- select items to drop here --*/
                if (start > 0) {     //  SKIPPING THE FIRST RECORD
                    subline[k] = '\0';
                    mtrx[jcol++] = atof(subline);
                }
                start = j + 1;
            }
        }
/*-------------------- next row */
        rlen = jcol - jcol0;
        jcol0 = jcol;
        if (lrow == 1) lfeat = rlen;
/*-------------------- inconsistent rows */
        if (rlen != lfeat) return (1);
    }
/*-------------------- done */
    fclose(finputs);
    //  for (j=0; j<jcol; j++)    printf(" %e  \n",mtrx[j]);
    *nrow = lrow;
    *nfeat = lfeat;
    return (0);
}

/*-------------------- assign a center to an item */

float dist2(float *x, float *y, int len) {
    int i;
    float dist = 0;
    for (i = 0; i < len; i++) {
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return dist / len;
}

/*=======================================================================*/

int MyKmeans_p(const float *inputDataIn, int *clustIdRet, int *counterRet, const int *params,
               float tolerance, MPI_Comm comm) {
/*==================================================
  IN: 
    inputData     = float* (featureCount*sampleCount)   = input data (local to this process).
    params[ ]  = int* contains 
    params[0] = clusterCount     = number of clusters
    params[1] = sampleCount      =  number of sampleCount
    params[2] = featureCount  = number of featureCount
    params[3] = maxIterations = max number of Kmeans iterations
    tolerance       = tolerance for determining if centers have converged.
    comm      = communicator
  OUT:
   clustId[i] = cluster of sample i for i=1...sampleCount
   counter[j] = size of cluster j for j=1:clusterCount
   ===================================================*/
    // some declarations
    int processCount, processId;
    /*-------------------- START */
    MPI_Comm_size(comm, &processCount);
    MPI_Comm_rank(comm, &processId);

    //-------------------- unpack params.
    int clusterCount = params[0];
    int sampleCount = params[1];
    printf("sampleCount %d\n", sampleCount);
    int featureCount = params[2];
    int maxIterations = params[3];

    float tolerance2 = tolerance * tolerance;
    //int NcNf =clusterCount*featureCount;
    /*-------------------- replace these by your function*/

    int data_size = featureCount * sampleCount;

    float inputData[data_size];

    for (int i = 0; i < data_size; ++i) {
        inputData[i] = inputDataIn[i];
    }

    // how much data each processor should process
    int chunkSize = (sampleCount / processCount);
    int sampleStart = chunkSize * processId;

    int sampleTo = sampleStart + chunkSize;
    int samplesLeftOver = sampleCount - sampleTo;

    if (samplesLeftOver < chunkSize) { // the last chunk is not large enough
        sampleTo = sampleCount;
    }

    assert(sampleStart >= 0);
    assert(sampleTo <= sampleCount);

    int center_size = featureCount * clusterCount;


    float old_centers[center_size];

    for (int i = 0; i < center_size; i++) {
        old_centers[i] = 0; // initialize at something
    }

    float centers[center_size];

    for (int clusterOn = 0; clusterOn < clusterCount; clusterOn++) {
        float *center = &centers[clusterOn * featureCount];
        get_rand_ftr(center, inputData, sampleCount, featureCount);
//        printf("center  %f\n", *center);
    }

    // sum all of the centers cross-process

    float updatedCenters[center_size];
    MPI_Allreduce(&centers, &updatedCenters, featureCount * clusterCount, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    memcpy(&centers, &updatedCenters, sizeof(centers));

    // average
    for (int i = 0; i < clusterCount * featureCount; i++) {
        centers[i] /= (float) processCount;
    }


    float sum[center_size];

    int iterOn = 0;

    int clustId[sampleCount];
    int counter[clusterCount];

    while (iterOn < maxIterations) {

        printf("sampleCount %d\n", sampleCount);

        for (int j = 0; j < sampleCount; j++) clustId[j] = 0;
        for (int j = 0; j < clusterCount; j++) counter[j] = 0;

        // reset sum
        for (int i = 0; i < featureCount * clusterCount; i++) sum[i] = 0;


        for (int sampleIdx = sampleStart; sampleIdx < sampleTo; sampleIdx++) {

            int dataStartIdx = sampleIdx * featureCount;

            int clusterMinIdx = -1;
            double dist2Min = DBL_MAX;

            // compute the closest cluster to the data point
            assert(dist2Min > 90000.0F);
            assert(clusterCount > 0);
            for (int clusterOn = 0; clusterOn < clusterCount; ++clusterOn) {

                int clusterStartIdx = featureCount * clusterOn;
                double dist2 = 0;

                // go over data from one sample
                for (int i = 0; i < featureCount; i++) {
                    int dataIdx = dataStartIdx + i;

                    assert(dataIdx >= 0);
                    assert(dataIdx < featureCount * sampleCount);

                    int clusterIdx = clusterStartIdx + i;
                    float on = inputData[dataIdx];
                    float expect = centers[clusterIdx];
                    double difference = on - expect;
                    double d2 = difference * difference;
                    assert__(d2 < 10000.0 * 10000.0){
                        printf("assert failed... on [%f], expeect[%f]\n", on, expect);
                    }
                    printf("difference %f\n", difference);
                    dist2 += d2;
                }

                assert(dist2 < DBL_MAX);

                if (dist2 <= dist2Min) {
                    dist2Min = dist2;
                    clusterMinIdx = clusterOn;
                }
            }

            assert(clusterMinIdx >= 0);


            // add the sample to the sum for the cluster
            for (int i = 0; i < featureCount; i++) {
                int dataIdx = dataStartIdx + i;

                assert(dataIdx >= 0);
                assert(dataIdx < featureCount * sampleCount);

                float datum = inputData[dataIdx];
                sum[clusterMinIdx * featureCount + i] += datum;
            }

            // change counters/clustIds accordingly
            counter[clusterMinIdx]++;
            clustId[sampleIdx] = clusterMinIdx;
        }

        // add sums and counters globally

        float updatedSum[center_size];
        MPI_Allreduce(&sum, &updatedSum, center_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        memcpy(&sum, &updatedSum, sizeof(sum));

        float updatedCounter[sampleCount];
        MPI_Allreduce(&counter, &updatedCounter, sampleCount, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        memcpy(&counter, &updatedCounter, sizeof(counter));

        // if the new data is within the threshold
        bool withinThreshold = true;

        // see if the data is within threshold and compute new centers
        for (int clusterOn = 0; clusterOn < clusterCount; clusterOn++) {

            // difference between old center and new center
            float difference2 = 0;

            int count = counter[clusterOn];
            float *sumStart = &sum[clusterOn * featureCount];
            float *centerStart = &centers[clusterOn * featureCount];


            // we need to sample
            if (count == 0) {
                printf("sum start\n");
                get_rand_ftr(sumStart, inputData, sampleCount, featureCount);

                // sum
                float sumStartUpdated[featureCount];
                MPI_Allreduce(&sumStart, &sumStartUpdated, featureCount, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                memcpy(&sumStart, &sumStartUpdated, sizeof(sumStart));
            }

            // average
            for (int i = 0; i < clusterCount * featureCount; i++) {
                float to = sumStart[i] / (float) processCount;

                centerStart[i] = to;

                float from = old_centers[i];
                float diff = from - to;
                float diff2 = diff * diff;
                difference2 += diff2;
            }

            if (difference2 > tolerance2) {
                withinThreshold = false;
            }
        }

        // we can "return" since we are within the threshold
        if (withinThreshold) {
            break;
        }

        // set old centers to current centers
        for (int i = 0; i < featureCount * sampleCount; i++) {
            old_centers[i] = centers[i];
        }

        iterOn++;
    }

    for (int j = 0; j < sampleCount; j++) {
        clustIdRet[j] = clustId[j];
    }
    for (int j = 0; j < clusterCount; j++) {
        int size = counter[j];
        printf("size %d\n", size);
        counterRet[j] = size;
    }

    printf("done!\n");


    return 0;
}
