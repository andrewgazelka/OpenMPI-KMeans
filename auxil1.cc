#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <memory>
#include "utils.h"
#include <limits>
#include "auxil1.h"


#define MAX_LINE 210

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

int MyKmeans_p(float *inputData, int *clustId, int *counter, const int *params,
               float tolerance, MPI_Comm comm) {
/*==================================================
  IN: 
    inputData     = float* (featureNum*sampleNum)   = input data (local to this process).
    params[ ]  = int* contains 
    params[0] = clusterNum     = number of clusters
    params[1] = sampleNum      =  number of sampleNum
    params[2] = featureNum  = number of featureNum
    params[3] = maxIterations = max number of Kmeans iterations
    tolerance       = tolerance for determining if centers have converged.
    comm      = communicator
  OUT:
   clustId[i] = cluster of sample i for i=1...sampleNum
   counter[j] = size of cluster j for j=1:clusterNum
   ===================================================*/
    // some declarations
    int processCount, processId;
    /*-------------------- START */
    MPI_Comm_size(comm, &processCount);
    MPI_Comm_rank(comm, &processId);

    //-------------------- unpack params.
    let clusterNum = params[0];
    let sampleNum = params[1];
    let featureNum = params[2];
    let maxIterations = params[3];

    let tolerance2 = tolerance * tolerance;
    //int NcNf =clusterNum*featureNum;
    /*-------------------- replace these by your function*/

    // how much data each processor should process
    let chunkSize = (sampleNum / processCount);
    let fromIdx = chunkSize * processId;

    auto sampleTo = fromIdx + chunkSize;
    let samplesLeftOver = sampleNum - sampleTo;

    if (samplesLeftOver < chunkSize) { // the last chunk is not large enough
        sampleTo = sampleNum;
    }

    std::unique_ptr<float[]> old_centers(new float[featureNum * sampleNum]{std::numeric_limits<float>::max()});
    std::unique_ptr<float[]> centers(new float[featureNum * sampleNum]{0});

    for (int clusterOn = 0; clusterOn < clusterNum; clusterOn++) {
        float *center = &centers[clusterOn * featureNum];
        get_rand_ftr(center, inputData, sampleNum, featureNum);
    }

    // sum all of the centers cross-process
    MPI_Allreduce(&centers, &centers, featureNum * clusterNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // average
    for (int i = 0; i < clusterNum * featureNum; i++) {
        centers[i] /= static_cast<float>(processCount);
    }


    std::unique_ptr<float[]> sum(new float[featureNum * clusterNum]);

    int iterOn = 0;

    while (iterOn < maxIterations) {

        for (int j = 0; j < sampleNum; j++) clustId[j] = 0;
        for (int j = 0; j < clusterNum; j++) counter[j] = 0;

        // reset sum
        for (int i = 0; i < featureNum * clusterNum; i++) sum[i] = 0;


        for (int sampleIdx = fromIdx; sampleIdx < sampleTo; sampleIdx++) {
            let dataStartIdx = sampleIdx * sampleNum;

            int clusterMinIdx = -1;
            float dist2Min = std::numeric_limits<float>::max();

            // compute the closest cluster to the data point
            for (int clusterOn = 0; clusterOn < clusterNum; ++clusterOn) {

                let clusterStartIdx = sampleNum * clusterOn;
                float dist2 = 0;

                // go over data from one sample
                for (int i = 0; i < featureNum; i++) {
                    let dataIdx = dataStartIdx + i;
                    let clusterIdx = clusterStartIdx + i;
                    let on = inputData[dataIdx];
                    let expect = centers[clusterIdx];
                    let difference = on - expect;
                    let d2 = difference * difference;
                    dist2 += d2;
                }
                if (dist2 < dist2Min) {
                    dist2Min = dist2;
                    clusterMinIdx = clusterOn;
                }
            }

            // add the sample to the sum for the cluster
            for (int i = 0; i < sampleNum; i++) {
                let dataIdx = dataStartIdx + i;
                sum[clusterMinIdx * featureNum + i] += inputData[dataIdx];
            }

            // change counters/clustIds accordingly
            counter[clusterMinIdx]++;
            clustId[sampleNum] = clusterMinIdx;
        }

        // add sums and counters globally
        MPI_Allreduce(&sum, &sum, clusterNum * sampleNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&counter, &counter, featureNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // if the new data is within the threshold
        bool withinThreshold = true;

        // see if the data is within threshold and compute new centers
        for (int clusterOn = 0; clusterOn < clusterNum; clusterOn++) {

            // difference between old center and new center
            float difference2 = 0;

            let count = counter[clusterOn];
            float *sumStart = &sum[clusterOn * featureNum];
            float *centerStart = &centers[clusterOn * featureNum];


            // we need to sample
            if (count == 0) {
                get_rand_ftr(sumStart, inputData, sampleNum, featureNum);

                // sum
                MPI_Allreduce(&sumStart, &sumStart, featureNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }

            // average
            for (int i = 0; i < clusterNum * featureNum; i++) {
                let from = old_centers[i];
                let to = sumStart[i] / static_cast<float>(processCount);;
                let diff = from - to;
                let diff2 = diff * diff;

                centerStart[i] = to;
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
        for (int i = 0; i < featureNum * sampleNum; i++){
            old_centers[i] = centers[i];
        }

        iterOn++;
    }


    return 0;
}
