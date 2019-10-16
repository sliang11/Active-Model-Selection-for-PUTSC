//On Linux
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
//#include <double.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ios>
#include <string>
#include <algorithm>
#include <vector>
#include "utilities.h"

#define INF 1e7
#define SQUARE (term1 - term2) * (term1 - term2)
#define MAX_CHAR_PER_LINE 200000

__device__ void getMuSigma(double &mu, double &sigma, double *ts, const int tsLen){
	double term, s, s2;
	s = s2 = 0;
	for (int i = 0; i < tsLen; i++){
		term = ts[i];
		s += term;
		s2 += term * term;
	}
	mu = s / tsLen;
	sigma = s2 / tsLen > mu * mu ? sqrt(s2 / tsLen - mu * mu) : 1;	//May cause inaccuracy due to loss of precision
}

__device__ void getNormalizedTerm(double &term, double mu, double sigma){
	term = (term - mu) / sigma;
}

__device__ void dtw(double &dist, double *ts, double *query, double *tmp,
	const int tsLen, const int maxWarp){

	double term1, term2, mu1, sigma1, mu2, sigma2;
	getMuSigma(mu1, sigma1, ts, tsLen);
	getMuSigma(mu2, sigma2, query, tsLen);

	if (tsLen == 1){
		term1 = ts[0];
		getNormalizedTerm(term1, mu1, sigma1);
		term2 = query[0];
		getNormalizedTerm(term2, mu2, sigma2);
		dist = sqrt(SQUARE);
	}
	else{
		int i;
		for (i = 0; i < 2 * (2 * maxWarp + 2); i++)
			tmp[i] = INF;

		term1 = ts[0];
		getNormalizedTerm(term1, mu1, sigma1);
		term2 = query[0];
		getNormalizedTerm(term2, mu2, sigma2);
		tmp[maxWarp] = SQUARE;

		int lowerRight = maxWarp < tsLen - 1 ? 2 * maxWarp : tsLen + maxWarp - 1;
		for (i = maxWarp + 1; i <= lowerRight; i++){
			term1 = ts[i - maxWarp];
			getNormalizedTerm(term1, mu1, sigma1);
			term2 = query[0];
			getNormalizedTerm(term2, mu2, sigma2);
			tmp[i] = tmp[i - 1] + SQUARE;
		}

		if (tsLen == 1)
			dist = sqrt(tmp[lowerRight]);
		else{
			double selected;
			int lower, upper, id, j, t, upperRight;
			lower = 0;
			upper = 2 * maxWarp + 2;
			j = 1;
			while (1){
				id = j - maxWarp;
				term2 = query[j];
				getNormalizedTerm(term2, mu2, sigma2);
				if (id < 0)
					tmp[upper + 1] = INF;
				else{
					selected = tmp[lower] < tmp[lower + 1] ? tmp[lower] : tmp[lower + 1];
					term1 = ts[id];
					getNormalizedTerm(term1, mu1, sigma1);
					tmp[upper + 1] = selected + SQUARE;
				}
				upperRight = maxWarp < tsLen - j ? 2 * maxWarp + 1 : tsLen + maxWarp - j;
				for (i = 2; i <= upperRight; i++){
					id = i + j - maxWarp - 1;
					if (id < 0)
						tmp[upper + i] = INF;
					else{
						selected = tmp[upper + i - 1] < tmp[lower + i] ? tmp[upper + i - 1] : tmp[lower + i];
						selected = selected < tmp[lower + i - 1] ? selected : tmp[lower + i - 1]; 
						term1 = ts[id];
						getNormalizedTerm(term1, mu1, sigma1);
						tmp[upper + i] = selected + SQUARE;
					}
				}

				if (j == tsLen - 1)
					break;

				t = lower;
				lower = upper;
				upper = t;

				for (i = 0; i < 2 * maxWarp + 1; i++)
					tmp[lower + i] = tmp[lower + i + 1];
				tmp[lower + 2 * maxWarp + 1] = INF;
				j++;
			}
			dist = sqrt(tmp[upper + upperRight]);
		}
	}
}

__global__ void getPDists(double *trainTss_in, double *testTss_in, double *pDists_out,
	const int numTrain, const int numTest, const int tsLen, const double bandwidth){

	int blockSize = blockDim.x;
	int numThreadsPerGrid = gridDim.x * blockSize;
	int numQueriesPerThread = ceil((double)numTest / numThreadsPerGrid);	//May be inaccurate due to loss of precision
	int maxWarp = ceil(tsLen * bandwidth);	//May be inaccurate due to loss of precision
	double *tmp = new double[2 * (2 * maxWarp + 2)];
	double *query, *ts;
	int i, j, next;
	for (i = 0; i < numQueriesPerThread; i++){
		next = (blockIdx.x * blockSize + threadIdx.x) * numQueriesPerThread + i;
		if (next >= numTest)
			break;
		query = &testTss_in[next * tsLen];
		for (j = 0; j < numTrain; j++){
			ts = &trainTss_in[j * tsLen];
			dtw(pDists_out[next * numTrain + j], ts, query, tmp, tsLen, maxWarp);
		}
	}
	delete tmp;
}

void saveResults(std::string dtwPath, std::string datasetName, std::string pfx,
	double bandwidth, int numTrain, int numTest, double *pDists){

	std::ostringstream os;
	os << bandwidth;
	std::string dtwFName = dtwPath + "/" + datasetName + "_dtw_" + pfx + "_" + os.str();
	if (access(dtwFName.c_str(), F_OK) == 0)
		remove(dtwFName.c_str());

	std::ofstream fout;
	fout.precision(15);
	fout.open(dtwFName.c_str(), std::ofstream::out | std::ofstream::app);
	for (int i = 0; i < numTest; i++){
		for (int j = 0; j < numTrain; j++)
			fout << pDists[i * numTrain + j] << " ";
		fout << std::endl;
	}
	fout.close();
}


int main(int argc, char **argv){

	if (argc < 4)
		exit(1);

	std::string datasetName = argv[1];
	const int numTrain = atoi(argv[2]);
	const int tsLen = atoi(argv[3]);
	const int deviceId = argc > 4 ? atoi(argv[4]) : 0;
	const int minIntWarp = argc > 5 ? atoi(argv[5]) : 0;
	const int maxIntWarp = argc > 6 ? atoi(argv[6]) : 20;
	const int intWarpStep = argc > 7 ? atoi(argv[7]) : 1;
	const int maxThreadsPerBlock = argc > 8 ? atoi(argv[8]) : 256;
	const int maxBlocksPerGrid = argc > 9 ? atoi(argv[9]) : 256;
	std::string datasetPath = argc > 10 ? argv[10] : "../Data";
	std::string dtwPath = argc > 11 ? argv[11] : "../Results/DTW";

	printf("%s\n", datasetName.c_str());

	cudaError_t cudaerr;
	cudaSetDevice(deviceId);
	
	//load data to host
	size_t trainTssBytes = numTrain * tsLen * sizeof(double);
	double *trainTss = (double*)malloc(trainTssBytes);
	size_t trainLabelsBytes = numTrain * sizeof(int);
	int *trainLabels = (int*)malloc(trainLabelsBytes);
	importTimeSeries(trainTss, trainLabels, datasetPath, datasetName, "TRAIN", numTrain, tsLen);
	free(trainLabels);

	//load data to device
	double *trainTss_in;
	cudaerr = cudaMalloc(&trainTss_in, trainTssBytes);
	if (cudaerr != cudaSuccess){
		printf("cudaMalloc for trainTss_in failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
	}
	cudaMemcpy(trainTss_in, trainTss, trainTssBytes, cudaMemcpyHostToDevice);
	free(trainTss);

	if (access(dtwPath.c_str(), F_OK) == -1)
		mkdir(dtwPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	int blockSize, gridSize;
	size_t pDistsBytes;
	double *pDists, *pDists_out;

	std::vector<double> bandwidths;
	for (int i_bandwidth = minIntWarp; i_bandwidth <= maxIntWarp; i_bandwidth += intWarpStep){
		bandwidths.push_back((double)i_bandwidth / 100);
	}
	
	double bandwidth;
	for (int i = 0; i < bandwidths.size(); i++){
		bandwidth = bandwidths[i];
		printf("bandwidth = %.3f absonluteWin = %d\n", bandwidth, (int)(ceil(tsLen * bandwidth)));

		//kernel launch
		pDistsBytes = numTrain * numTrain * sizeof(double);
		cudaMalloc(&pDists_out, pDistsBytes);
		blockSize = numTrain < maxThreadsPerBlock ? numTrain : maxThreadsPerBlock;
		gridSize = ceil((double)numTrain / blockSize) < maxBlocksPerGrid ? ceil((double)numTrain / blockSize) : maxBlocksPerGrid;
		getPDists << <gridSize, blockSize >> > (trainTss_in, trainTss_in, pDists_out, numTrain, numTrain, tsLen, bandwidth);

		cudaerr = cudaThreadSynchronize();
		if (cudaerr != cudaSuccess){
			printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
			exit(1);
		}

		//load results to host
		pDists = (double *)malloc(pDistsBytes);
		cudaMemcpy(pDists, pDists_out, pDistsBytes, cudaMemcpyDeviceToHost);
		cudaFree(pDists_out);

		//save results
		saveResults(dtwPath, datasetName, "TRAIN", bandwidth, numTrain, numTrain, pDists);
		free(pDists);
	}
	cudaFree(trainTss_in);
	return 0;
}