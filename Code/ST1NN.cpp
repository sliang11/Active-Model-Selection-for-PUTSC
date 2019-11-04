#include "utilities.h"
#include "evaluationUtilities.h"
#include <vector>
#include <tuple>
#include <sstream>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <unistd.h>
#include <sys/stat.h>

#define INF 1e6
#define MAX_CHAR 10
#define MAX_CHAR_PER_LINE 200000

void getPDists_DTWD(double* pDistMtx_DTWD, double* pDistMtx_ED, double* pDistMtx_DTW, int numTrain) {
	for (int i = 0; i < numTrain; i++) {
		pDistMtx_DTWD[i * numTrain + i] = INF;
		for (int j = i + 1; j < numTrain; j++) {

			double dist;
			if (doubleIsZero(pDistMtx_ED[i * numTrain + j])) {
				dist = 0;
			}
			else {
				dist = pDistMtx_DTW[i * numTrain + j] / pDistMtx_ED[i * numTrain + j];
			}
			pDistMtx_DTWD[i * numTrain + j] = pDistMtx_DTWD[j * numTrain + i] = dist;
		}
	}
}

void getNNs(double* nnDists, int* nnInds, double* pDistMtx, int numTrain) {
	double* pDistVec;
	for (int i = 0; i < numTrain; i++) {
		pDistVec = pDistMtx + i * numTrain;
		min(nnDists[i], nnInds[i], pDistVec, numTrain);
	}
}

bool checkNNIsRanked(int* rankedInds, int nnInd, int numRanked) {
	for (int i = 0; i < numRanked; i++) {
		if (nnInd == rankedInds[i])
			return true;
	}
	return false;
}

void update(int* rankedInds, double* minNNDists, int* nnPInds, double* pDistMtx, bool* ranked, int numTrain, int numRanked, int numPLabeled) {
	double* pDistVec, nnDistU, dist, minNNDist = INF;
	int curInd, nnIndU, minNNInd;
	int nnPInd;
	for (int i = 0; i < numRanked; i++) {
		curInd = rankedInds[i];
		pDistVec = pDistMtx + curInd * numTrain;

		nnDistU = INF;
		for (int j = 0; j < numTrain; j++) {
			if (ranked[j])
				continue;

			dist = pDistVec[j];
			if (dist < nnDistU) {
				nnIndU = j;
				nnDistU = dist;
			}
		}

		if (nnDistU < minNNDist) {
			minNNInd = nnIndU;
			minNNDist = nnDistU;
			nnPInd = curInd;
		}
	}
	rankedInds[numRanked] = minNNInd;
	minNNDists[numRanked - numPLabeled] = minNNDist;
	nnPInds[numRanked - numPLabeled] = nnPInd;
	ranked[minNNInd] = true;
}

void rankTrainInds(int* rankedInds, double* minNNDists, int* nnPInds, int* seed, double* pDistMtx, bool* ranked, int numTrain, int numPLabeled) {

	memcpy(rankedInds, seed, numPLabeled * sizeof(int));
	memset(ranked, 0, numTrain * sizeof(bool));
	for (int i = 0; i < numTrain; i++) {
		for (int j = 0; j < numPLabeled; j++) {
			if (i == seed[j]) {
				ranked[i] = true;
				break;
			}
		}
	}
	for (int i = numPLabeled; i < numTrain; i++) {
		update(rankedInds, minNNDists, nnPInds, pDistMtx, ranked, numTrain, i, numPLabeled);
	}

}

void getPrfsByIter(double* precisions, double* recalls, double* fscores,
	int* rankedInds, int* realLabels, int* preLabels, int numTrain, int numPLabeled) {

	memset(preLabels, 0, numTrain * sizeof(int));
	for (int i = 0; i < numTrain; i++) {
		preLabels[rankedInds[i]] = 1;
		if (i >= numPLabeled) {
			prfWithSeed(precisions[i - numPLabeled], recalls[i - numPLabeled], fscores[i - numPLabeled],
				realLabels, preLabels, rankedInds, numTrain, numPLabeled);
		}
	}
}

void update_WK(int* rankedInds_WK, double* minNNDists_WK, int* nnPInds_WK, double* pDistMtx, bool* ranked_WK, bool* validU_WK, int numTrain, int numRanked, int numPLabeled) {
	double* pDistVec, nnDistU, dist, minNNDist = INF;
	int curInd, nnIndU, minNNInd;
	int nnPInd;
	for (int i = 0; i < numRanked; i++) {
		curInd = rankedInds_WK[i];
		pDistVec = pDistMtx + curInd * numTrain;

		nnDistU = INF;
		for (int j = 0; j < numTrain; j++) {
			if (!validU_WK[j])
				continue;

			dist = pDistVec[j];
			if (dist < nnDistU) {
				nnIndU = j;
				nnDistU = dist;
			}
		}

		if (nnDistU < minNNDist) {
			minNNInd = nnIndU;
			minNNDist = nnDistU;
			nnPInd = curInd;
		}
	}
	rankedInds_WK[numRanked] = minNNInd;
	minNNDists_WK[numRanked - numPLabeled] = minNNDist;
	nnPInds_WK[numRanked - numPLabeled] = nnPInd;
	ranked_WK[minNNInd] = true;
}

//Li Wei, Eamonn J.Keogh: Semi-supervised time series classification. KDD 2006: 748-753
//This is NOT an accurate re-implementation of WK, and is NOT used in the experiments.
int rankTrainInds_WK(int* rankedInds_WK, double* minNNDists_WK, int* nnPInds_WK, double* nnDists, int* nnInds, int* seed,
	double* pDistMtx, bool* ranked_WK, bool* validU_WK, int numTrain, int numPLabeled, int minNumP, int maxNumP) {

	memcpy(rankedInds_WK, seed, numPLabeled * sizeof(int));
	memset(ranked_WK, 0, numTrain * sizeof(bool));
	for (int i = 0; i < numTrain; i++) {
		for (int j = 0; j < numPLabeled; j++) {
			if (i == seed[j]) {
				ranked_WK[i] = true;
				break;
			}
		}
	}

	getNNs(nnDists, nnInds, pDistMtx, numTrain);
	int numValidU;
	for (int i = numPLabeled; i < maxNumP; i++) {
		memset(validU_WK, 0, numTrain * sizeof(bool));
		numValidU = 0;
		for (int j = 0; j < numTrain; j++) {
			if (ranked_WK[j] || !ranked_WK[nnInds[j]])
				continue;
			validU_WK[j] = true;
			numValidU++;
		}

		if (!numValidU && i >= minNumP) {
			return i;
		}
		else {
			if (!numValidU) {
				for (int j = 0; j < numTrain; j++) {
					validU_WK[j] = !ranked_WK[j];
				}
			}
			update_WK(rankedInds_WK, minNNDists_WK, nnPInds_WK, pDistMtx, ranked_WK, validU_WK, numTrain, i, numPLabeled);
		}
	}
	return maxNumP;

}

//Chotirat Ann Ratanamahatana, Dechawut Wanichsan: Stopping Criterion Selection for Efficient Semi-supervised Time Series Classification.Software Engineering, Artificial Intelligence, Networking and Parallel / Distributed Computing 2008: 1-14
int sc_RW(double* minNNDists, int minNumP, int maxNumP, int numTrain, int numPLabeled) {

	//initialization
	double minNNDist, sum, sum2;
	sum = sum2 = 0;
	for (int i = 0; i < minNumP - numPLabeled + 1; i++) {
		minNNDist = minNNDists[i];
		sum += minNNDist;
		sum2 += minNNDist * minNNDist;
	}

	double diff, std, scc, maxScc = -INF;
	int preNumP, initNumU = numTrain - numPLabeled;
	for (int i = minNumP - numPLabeled + 1; i < min(maxNumP + 2, numTrain) - numPLabeled; i++) {
		minNNDist = minNNDists[i];
		sum += minNNDist;
		sum2 += minNNDist * minNNDist;

		diff = fabs(minNNDists[i] - minNNDists[i - 1]);
		std = sum2 / (i + 1) - sum * sum / ((i + 1) * (i + 1));
		std = !doubleLeq(std, 0.0) ? sqrt(std) : 1;
		scc = diff / std * (double)(initNumU - i) / initNumU;

		if (scc > maxScc) {
			maxScc = scc;
			preNumP = numPLabeled + i - 1;
		}
	}
	return preNumP;
}

void discretize(int* seq, double* ts, int tsLen, int card) {

	double minVal = min(ts, tsLen);
	double maxVal = max(ts, tsLen);
	if (doubleIsEqual(minVal, maxVal)) {
		for (int i = 0; i < tsLen; i++) {
			seq[i] = 1;
		}
	}
	else {
		for (int i = 0; i < tsLen; i++) {
			seq[i] = round((ts[i] - minVal) / (maxVal - minVal) * (card - 1)) + 1;	//this can lead to incorrect values due to loss of precision
		}
	}

}

long long getRdl(int* hypoSeq, long long& cumNumMiss, double* nextTs, int* nextSeq, int numTrain, int numRanked, int tsLen, int card) {
	discretize(nextSeq, nextTs, tsLen, card);
	for (int i = 0; i < tsLen; i++) {
		if (nextSeq[i] != hypoSeq[i])
			cumNumMiss++;
	}

	//needs log2(card) to be an integer
	return (double)(numTrain - numRanked + 1) * tsLen * log2(card)
		+ cumNumMiss * (log2(card) + ceil(log2(tsLen)));	//May require fixing! Can this lead to incorrect results due to loss of precision?
}

//Nurjahan Begum, Bing Hu, Thanawin Rakthanmanon, Eamonn J.Keogh: A Minimum Description Length Technique for Semi-Supervised Time Series Classification. IRI 2013: 171 - 192
int sc_BHRK(double* tss, int* rankedInds, int* hypoSeq, int* nextSeq,
	int minNumP, int maxNumP, int numTrain, int numPLabeled, int tsLen, int card) {

	double* ts;
	int preNumP, optPreNumP;
	long long cumNumMiss, curRdl, prevRdl, minRdl = LLONG_MAX;
	for (int i = 0; i < numPLabeled; i++) {
		ts = tss + rankedInds[i] * tsLen;
		discretize(hypoSeq, ts, tsLen, card);
		prevRdl = LLONG_MAX;
		cumNumMiss = 0;
		preNumP = 0;
		for (int j = numPLabeled; j < maxNumP; j++) {
			ts = tss + rankedInds[j] * tsLen;
			curRdl = getRdl(hypoSeq, cumNumMiss, ts, nextSeq, numTrain, j + 1, tsLen, card);

			if (j < minNumP || curRdl < prevRdl) {
				prevRdl = curRdl;
			}
			else {
				preNumP = j;
				break;
			}
		}
		if (!preNumP)
			preNumP = maxNumP;
		if (prevRdl < minRdl) {
			minRdl = prevRdl;
			optPreNumP = preNumP;
		}
	}
	return optPreNumP;
}

//This implementation cannot handle cases where there are consecutive identical values in minNNDists correctly. However, we believe such cases are rare.
//Also, in this implementation, overlapping intervals are not allowed, except that the finishing point of one can be the starting point of the next.
std::vector<std::tuple<int, int, int, int>> getIntervals(double* minNNDists, int numTrain, int numPLabeled, double beta) {

	std::vector<std::tuple<int, int, int, int>> intervals;	//start, ad, ds, finish
	int start, ad, ds, finish, instTrend, prevTrend, curTrend;
	prevTrend = 0; curTrend = -1; //1 for ascend, -1 for descend, 0 for stable
	double minNNDist, prevMinNNDist, diff, hd, lb, ub;
	prevMinNNDist = minNNDists[0];
	hd = INF; lb = INF; ub = -INF;
	start = ad = ds = finish = INF;
	for (int i = 1; i < numTrain - numPLabeled; i++) {

		minNNDist = minNNDists[i];
		if (!doubleIsEqual(minNNDist, lb) && !doubleIsEqual(minNNDist, ub) &&
			minNNDist > lb && minNNDist < ub) {
			instTrend = 0;
		}
		else {
			diff = minNNDist - prevMinNNDist;
			instTrend = sign(diff);
			if (!instTrend) {
				instTrend = curTrend;
			}
		}

		if (curTrend != instTrend) {
			curTrend = instTrend;
		}

		if (prevTrend == 0) {
			if (curTrend == 0) {

				if (i == numTrain - numPLabeled - 1) {
					finish = i;
					if (start < ad && ad < ds && ds < finish) {
						intervals.push_back(std::make_tuple(start, ad, ds, finish));
					}
				}
			}
			else if (curTrend == 1) {
				finish = i - 1;
				if (start < ad && ad < ds && ds < finish) {
					intervals.push_back(std::make_tuple(start, ad, ds, finish));
				}
				start = i - 1;
				hd = INF; lb = INF; ub = -INF;
			}
			else {
				finish = i - 1;
				if (start < ad && ad < ds && ds < finish) {
					intervals.push_back(std::make_tuple(start, ad, ds, finish));
				}
				hd = INF; lb = INF; ub = -INF;
			}
		}
		else if (prevTrend == 1) {
			if (curTrend == -1) {
				ad = i - 1;
				hd = -diff;
			}
		}
		else {
			if (curTrend == 0) {

				//This case is impossible.
			}
			else if (curTrend == 1) {

				if (hd != INF) {
					lb = prevMinNNDist - beta * hd;
					ub = prevMinNNDist + beta * hd;
				}
				else {
					lb = INF; ub = -INF;
				}
				if (!doubleIsEqual(minNNDist, lb) && !doubleIsEqual(minNNDist, ub) &&
					minNNDist > lb && minNNDist < ub) {
					curTrend = 0;
					ds = i - 1;

					if (i == numTrain - numPLabeled - 1) {
						finish = i;
						if (start < ad && ad < ds && ds < finish) {
							intervals.push_back(std::make_tuple(start, ad, ds, finish));
						}
					}

				}
				else {
					start = i - 1;
					hd = INF; lb = INF; ub = -INF;
				}

			}
			else {
				if (hd != INF) {
					hd -= diff;
				}
			}
		}
		prevTrend = curTrend;
		prevMinNNDist = minNNDist;
	}
	return intervals;
}

//Mabel Gonz¨￠lez Castellanos, Christoph Bergmeir, Isaac Triguero, Yanet Rodr¨aguez, Jos¨| Manuel Ben¨atez: On the stopping criteria for k - Nearest Neighbor in positive unlabeled time series classification problems. Inf.Sci. 328: 42-59 (2016)
void sc_GBTRM(int* preNumPs, double* minNNDists, int minNumP, int maxNumP, int numTrain, int numPLabeled, double beta) {

	int initNumU = numTrain - numPLabeled;
	double max_minNNDists = max(minNNDists, initNumU);

	int start, finish, ad, ds, ip, ws, preNumP;
	double maxScs[5], scs[5], ha, hd, max_interval, lw;
	for (int i = 0; i < 5; i++)
		preNumPs[i] = maxScs[i] = -INF;

	std::tuple<int, int, int, int> curInterval;
	std::vector<std::tuple<int, int, int, int>> intervals = getIntervals(minNNDists, numTrain, numPLabeled, beta);
	for (int i = 0; i < intervals.size(); i++) {

		curInterval = intervals[i];
		start = std::get<0>(curInterval);
		finish = std::get<3>(curInterval);
		ad = std::get<1>(curInterval);
		ds = std::get<2>(curInterval);

		ha = minNNDists[ad] - minNNDists[start];
		hd = minNNDists[ad] - minNNDists[ds];
		ws = finish - ds;
		max(max_interval, ip, minNNDists + start, finish - start + 1);
		ip += start;
		lw = (double)(initNumU - ip) / initNumU;

		scs[0] = hd * lw;
		scs[1] = ha * lw;
		scs[2] = ws * lw;
		scs[3] = max(ha, hd) * lw;
		scs[4] = max(hd / max_minNNDists, (double)ws / (initNumU - 1)) * lw;
		for (int j = 0; j < 5; j++) {
			if (scs[j] > maxScs[j]) {
				preNumP = numPLabeled + ip;
				if (preNumP >= minNumP && preNumP <= maxNumP) {
					maxScs[j] = scs[j];
					preNumPs[j] = preNumP;
				}
			}
		}
	}
}

void getTransPreLabels(int* transPreLabels, int* rankedInds, int numTrain, int preNumP) {

	if (preNumP < 0) {
		for (int i = 0; i < numTrain; i++) {
			transPreLabels[i] = -1;
		}
	}
	else {
		memset(transPreLabels, 0, numTrain * sizeof(int));
		for (int i = 0; i < preNumP; i++) {
			transPreLabels[rankedInds[i]] = 1;
		}
	}

}

int main(int argc, char** argv) {

	//parameter settings
	if (argc < 5) {
		printf("Exiting due to too few input arguments.\n");
		exit(1);
	}
	const std::string datasetName = argv[1];
	const int numTrain = atoi(argv[2]);
	const int numP = atoi(argv[3]);
	const int tsLen = atoi(argv[4]);
	const int minIntWarp = argc > 5 ? atoi(argv[5]) : 0;	//Do NOT change this setting!
	const int maxIntWarp = argc > 6 ? atoi(argv[6]) : 20;	//Do NOT change this setting!
	const int intWarpStep = argc > 7 ? atoi(argv[7]) : 1;	//Do NOT change this setting!
	const int minNumIters = argc > 8 ? atoi(argv[8]) : 1;	//Do NOT change this unless you know what you are doing!
	const int maxNumIters = argc > 9 ? atoi(argv[9]) : numTrain - 2; //Do NOT change this unless you know what you are doing!
	const std::string datasetPath = argc > 10 ? argv[10] : "../Data";
	const std::string dtwPath = argc > 11 ? argv[11] : "../Results/DTW";
	const std::string seedPath = argc > 12 ? argv[12] : "../Seeds/ST1NN";
	const std::string outputPath = argc > 13 ? argv[13] : "../Results/ST1NN";

	const int cards[]{ 4, 8, 16, 32, 64 };
	const double betas[]{ 0.1, 0.2, 0.3, 0.4, 0.5 };

	const int numSeeds = min(numP, 10);

	std::string fName;
	
	if (access(outputPath.c_str(), F_OK) == -1)
		mkdir(outputPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	//load time series
	size_t trainTssBytes = numTrain * tsLen * sizeof(double);
	double* trainTss = (double*)malloc(trainTssBytes);
	size_t trainLabelsBytes = numTrain * sizeof(int);
	int* trainLabels = (int*)malloc(trainLabelsBytes);
	importTimeSeries(trainTss, trainLabels, datasetPath, datasetName, "TRAIN", numTrain, tsLen);
	relabel(trainLabels, numTrain, 1);

	//load seeds
	const int numPLabeled = 1;
	char s_numSeeds[MAX_CHAR], s_numPLabeled[MAX_CHAR];
	sprintf(s_numSeeds, "%d", numSeeds);
	sprintf(s_numPLabeled, "%d", numPLabeled);
	int* seeds = (int*)malloc(numSeeds * numPLabeled * sizeof(int));
	fName = seedPath + "/seeds_" + datasetName + "_" + s_numSeeds + "_" + s_numPLabeled + ".txt";
	importMatrix(seeds, fName, numSeeds, numPLabeled, 1);
	for (int i = 0; i < numSeeds * numPLabeled; i++)
		seeds[i]--;	//matlab -> c
	int* seed;

	//DTW warping window
	std::vector<int> v_intWarps;
	int intWarp;
	for (intWarp = minIntWarp; intWarp <= maxIntWarp; intWarp += intWarpStep) {
		v_intWarps.push_back(intWarp);
	}

	int minNumP = minNumIters + numPLabeled;
	int maxNumP = maxNumIters + numPLabeled;

	int* rankedInds = (int*)malloc(numTrain * sizeof(int));
	double* minNNDists = (double*)malloc((numTrain - numPLabeled) * sizeof(double));
	bool* ranked = (bool*)malloc(numTrain * sizeof(bool));
	int* nnPInds = (int*)malloc((numTrain - numPLabeled) * sizeof(int));

	double* precisions = (double*)malloc((numTrain - numPLabeled) * sizeof(double));
	double* recalls = (double*)malloc((numTrain - numPLabeled) * sizeof(double));
	double* fscores = (double*)malloc((numTrain - numPLabeled) * sizeof(double));
	int* preLabels = (int*)malloc(numTrain * sizeof(int));

	//////WK
	double* nnDists = (double*)malloc(numTrain * sizeof(double));
	int* nnInds = (int*)malloc(numTrain * sizeof(int));
	int* rankedInds_WK = (int*)malloc(numTrain * sizeof(int));
	memset(rankedInds_WK, 0, numTrain * sizeof(int));
	double* minNNDists_WK = (double*)malloc((numTrain - numPLabeled) * sizeof(double));
	bool* ranked_WK = (bool*)malloc(numTrain * sizeof(bool));
	bool* validU_WK = (bool*)malloc(numTrain * sizeof(bool));
	double* precisions_WK = (double*)malloc((numTrain - numPLabeled) * sizeof(double));
	double* recalls_WK = (double*)malloc((numTrain - numPLabeled) * sizeof(double));
	double* fscores_WK = (double*)malloc((numTrain - numPLabeled) * sizeof(double));
	int* preLabels_WK = (int*)malloc(numTrain * sizeof(int));
	int* nnPInds_WK = (int*)malloc((numTrain - numPLabeled) * sizeof(int));

	size_t pDistBytes = numTrain * numTrain * sizeof(double);
	double* pDistMtx_DTWD = (double*)malloc(pDistBytes);

	int* hypoSeq = (int*)malloc(tsLen * sizeof(int));
	int* nextSeq = (int*)malloc(tsLen * sizeof(int));

	//load distance matrices
	int numWarps = v_intWarps.size();
	double warp, * pDistMtx_DTW, * pDistMtx_ED;
	double** allPDistMtx_DTW = safeMalloc<double*>(numWarps);
	for (int i = 0; i < numWarps; i++) {
		warp = (double)v_intWarps[i] / 100;

		allPDistMtx_DTW[i] = safeMalloc<double>(numTrain * numTrain);
		pDistMtx_DTW = allPDistMtx_DTW[i];

		std::ostringstream os;
		os << warp;
		fName = dtwPath + "/" + datasetName + "_dtw_TRAIN_" + os.str();
		importMatrix(pDistMtx_DTW, fName, numTrain, numTrain, 0);
		for (int j = 0; j < numTrain; j++)
			pDistMtx_DTW[j * numTrain + j] = INF;
	}
	pDistMtx_ED = allPDistMtx_DTW[0];	//set ED

	//output
	int* rankedIndsAll_DTW = (int*)malloc(v_intWarps.size() * numTrain * sizeof(int));
	int* rankedIndsAll_DTWD = (int*)malloc(v_intWarps.size() * numTrain * sizeof(int));
	int* rankedIndsAll_WK_DTW = (int*)malloc(v_intWarps.size() * numTrain * sizeof(int));
	int* rankedIndsAll_WK_DTWD = (int*)malloc(v_intWarps.size() * numTrain * sizeof(int));
	int* preNumPsAll_DTW = (int*)malloc(v_intWarps.size() * 33 * sizeof(int));
	int* preNumPsAll_DTWD = (int*)malloc(v_intWarps.size() * 33 * sizeof(int));
	double* fscoresAll_DTW = (double*)malloc(v_intWarps.size() * 33 * sizeof(double));
	double* fscoresAll_DTWD = (double*)malloc(v_intWarps.size() * 33 * sizeof(double));
	int* nnPIndsAll_DTW = (int*)malloc(v_intWarps.size() * (numTrain - numPLabeled) * sizeof(int));
	int* nnPIndsAll_DTWD = (int*)malloc(v_intWarps.size() * (numTrain - numPLabeled) * sizeof(int));
	int* nnPIndsAll_WK_DTW = (int*)malloc(v_intWarps.size() * (numTrain - numPLabeled) * sizeof(int));
	int* nnPIndsAll_WK_DTWD = (int*)malloc(v_intWarps.size() * (numTrain - numPLabeled) * sizeof(int));

	double fscore;
	int preNumP, preNumPs[5], localIdx;

	char s_seedId[MAX_CHAR];
	for (int seedId = 0; seedId < numSeeds; seedId++) {
		std::cout << "seedId = " << seedId << std::endl;
		sprintf(s_seedId, "%d", seedId);
		seed = seeds + seedId * numPLabeled;

		for (int i = 0; i < v_intWarps.size(); i++) {

			//DTW
			pDistMtx_DTW = allPDistMtx_DTW[i];

			//get rankings
			rankTrainInds(rankedInds, minNNDists, nnPInds, seed, pDistMtx_DTW, ranked, numTrain, numPLabeled);
			memcpy(rankedIndsAll_DTW + i * numTrain, rankedInds, numTrain * sizeof(int));
			memcpy(nnPIndsAll_DTW + i * (numTrain - numPLabeled), nnPInds, (numTrain - numPLabeled) * sizeof(int));
			getPrfsByIter(precisions, recalls, fscores, rankedInds, trainLabels, preLabels, numTrain, numPLabeled);

			//////Oracle
			localIdx = 0;
			preNumP = numP;
			fscore = fscores[preNumP - numPLabeled - 1];
			preNumPsAll_DTW[i * 33 + localIdx] = preNumP;
			fscoresAll_DTW[i * 33 + localIdx] = fscore;
			localIdx++;

			//////WK
			preNumP = rankTrainInds_WK(rankedInds_WK, minNNDists_WK, nnPInds_WK, nnDists, nnInds, seed,
				pDistMtx_DTW, ranked_WK, validU_WK, numTrain, numPLabeled, minNumP, maxNumP);
			memcpy(rankedIndsAll_WK_DTW + i * numTrain, rankedInds_WK, numTrain * sizeof(int));
			memcpy(nnPIndsAll_WK_DTW + i * (numTrain - numPLabeled), nnPInds_WK, (numTrain - numPLabeled) * sizeof(int));
			getPrfsByIter(precisions_WK, recalls_WK, fscores_WK, rankedInds_WK, trainLabels, preLabels_WK, numTrain, numPLabeled);
			fscore = fscores_WK[preNumP - numPLabeled - 1];
			preNumPsAll_DTW[i * 33 + localIdx] = preNumP;
			fscoresAll_DTW[i * 33 + localIdx] = fscore;
			localIdx++;

			//////RW
			preNumP = sc_RW(minNNDists, minNumP, maxNumP, numTrain, numPLabeled);
			fscore = fscores[preNumP - numPLabeled - 1];
			preNumPsAll_DTW[i * 33 + localIdx] = preNumP;
			fscoresAll_DTW[i * 33 + localIdx] = fscore;
			localIdx++;

			//////BHRK
			for (int j = 0; j < 5; j++) {
				preNumP = sc_BHRK(trainTss, rankedInds, hypoSeq, nextSeq, minNumP, maxNumP, numTrain, numPLabeled, tsLen, cards[j]);
				fscore = fscores[preNumP - numPLabeled - 1];
				preNumPsAll_DTW[i * 33 + localIdx] = preNumP;
				fscoresAll_DTW[i * 33 + localIdx] = fscore;
				localIdx++;
			}

			//////GBTRM
			for (int k = 0; k < 5; k++) {
				sc_GBTRM(preNumPs, minNNDists, minNumP, maxNumP, numTrain, numPLabeled, betas[k]);
				for (int j = 0; j < 5; j++) {
					if (preNumPs[j] < 0) {
						fscore = -2;
					}
					else {
						fscore = fscores[preNumPs[j] - numPLabeled - 1];
					}
					preNumPsAll_DTW[i * 33 + localIdx] = preNumPs[j];
					fscoresAll_DTW[i * 33 + localIdx] = fscore;
					localIdx++;
				}
			}

			//DTW-D
			getPDists_DTWD(pDistMtx_DTWD, pDistMtx_ED, pDistMtx_DTW, numTrain);
			rankTrainInds(rankedInds, minNNDists, nnPInds, seed, pDistMtx_DTWD, ranked, numTrain, numPLabeled);
			memcpy(rankedIndsAll_DTWD + i * numTrain, rankedInds, numTrain * sizeof(int));
			memcpy(nnPIndsAll_DTWD + i * (numTrain - numPLabeled), nnPInds, (numTrain - numPLabeled) * sizeof(int));
			getPrfsByIter(precisions, recalls, fscores, rankedInds, trainLabels, preLabels, numTrain, numPLabeled);

			//////Oracle
			localIdx = 0;
			preNumP = numP;
			fscore = fscores[preNumP - numPLabeled - 1];
			preNumPsAll_DTWD[i * 33 + localIdx] = preNumP;
			fscoresAll_DTWD[i * 33 + localIdx] = fscore;
			localIdx++;

			//////WK
			preNumP = rankTrainInds_WK(rankedInds_WK, minNNDists_WK, nnPInds_WK, nnDists, nnInds, seed,
				pDistMtx_DTWD, ranked_WK, validU_WK, numTrain, numPLabeled, minNumP, maxNumP);
			memcpy(rankedIndsAll_WK_DTWD + i * numTrain, rankedInds_WK, numTrain * sizeof(int));
			memcpy(nnPIndsAll_WK_DTWD + i * (numTrain - numPLabeled), nnPInds_WK, (numTrain - numPLabeled) * sizeof(int));
			getPrfsByIter(precisions_WK, recalls_WK, fscores_WK, rankedInds_WK, trainLabels, preLabels_WK, numTrain, numPLabeled);
			fscore = fscores_WK[preNumP - numPLabeled - 1];
			preNumPsAll_DTWD[i * 33 + localIdx] = preNumP;
			fscoresAll_DTWD[i * 33 + localIdx] = fscore;
			localIdx++;

			//////RW
			preNumP = sc_RW(minNNDists, minNumP, maxNumP, numTrain, numPLabeled);
			fscore = fscores[preNumP - numPLabeled - 1];
			preNumPsAll_DTWD[i * 33 + localIdx] = preNumP;
			fscoresAll_DTWD[i * 33 + localIdx] = fscore;
			localIdx++;

			//////BHRK
			for (int j = 0; j < 5; j++) {
				preNumP = sc_BHRK(trainTss, rankedInds, hypoSeq, nextSeq, minNumP, maxNumP, numTrain, numPLabeled, tsLen, cards[j]);
				fscore = fscores[preNumP - numPLabeled - 1];
				preNumPsAll_DTWD[i * 33 + localIdx] = preNumP;
				fscoresAll_DTWD[i * 33 + localIdx] = fscore;
				localIdx++;
			}

			/////GBTRM
			for (int k = 0; k < 5; k++) {
				sc_GBTRM(preNumPs, minNNDists, minNumP, maxNumP, numTrain, numPLabeled, betas[k]);
				for (int j = 0; j < 5; j++) {
					if (preNumPs[j] < 0) {
						fscore = -2;
					}
					else {
						fscore = fscores[preNumPs[j] - numPLabeled - 1];
					}
					preNumPsAll_DTWD[i * 33 + localIdx] = preNumPs[j];
					fscoresAll_DTWD[i * 33 + localIdx] = fscore;
					localIdx++;
				}
			}

		}

		//Save to file
		std::ofstream fout;
		fName = outputPath + "/" + datasetName + "_rankedInds_DTW_rand10_single_" + s_seedId + ".txt";
		fout.open(fName.c_str());
		for (int i = 0; i < v_intWarps.size(); i++) {
			for (int j = 0; j < numTrain; j++) {
				fout << rankedIndsAll_DTW[i * numTrain + j] << " ";
			}
			fout << std::endl;
		}
		fout.close();

		fName = outputPath + "/" + datasetName + "_rankedInds_DTWD_rand10_single_" + s_seedId + ".txt";
		fout.open(fName.c_str());
		for (int i = 0; i < v_intWarps.size(); i++) {
			for (int j = 0; j < numTrain; j++) {
				fout << rankedIndsAll_DTWD[i * numTrain + j] << " ";
			}
			fout << std::endl;
		}
		fout.close();

		fName = outputPath + "/" + datasetName + "_preNumPs_DTW_rand10_single_" + s_seedId + ".txt";
		fout.open(fName.c_str());
		for (int i = 0; i < v_intWarps.size(); i++) {
			for (int j = 0; j < 33; j++) {
				fout << preNumPsAll_DTW[i * 33 + j] << " ";
			}
			fout << std::endl;
		}
		fout.close();

		fName = outputPath + "/" + datasetName + "_preNumPs_DTWD_rand10_single_" + s_seedId + ".txt";
		fout.open(fName.c_str());
		for (int i = 0; i < v_intWarps.size(); i++) {
			for (int j = 0; j < 33; j++) {
				fout << preNumPsAll_DTWD[i * 33 + j] << " ";
			}
			fout << std::endl;
		}
		fout.close();

		fout.precision(15);
		fName = outputPath + "/" + datasetName + "_transFscores_DTW_rand10_single_" + s_seedId + ".txt";
		fout.open(fName.c_str());
		for (int i = 0; i < v_intWarps.size(); i++) {
			for (int j = 0; j < 33; j++) {
				fout << fscoresAll_DTW[i * 33 + j] << " ";
			}
			fout << std::endl;
		}
		fout.close();

		fName = outputPath + "/" + datasetName + "_transFscores_DTWD_rand10_single_" + s_seedId + ".txt";
		fout.open(fName.c_str());
		for (int i = 0; i < v_intWarps.size(); i++) {
			for (int j = 0; j < 33; j++) {
				fout << fscoresAll_DTWD[i * 33 + j] << " ";
			}
			fout << std::endl;
		}
		fout.close();

		fName = outputPath + "/" + datasetName + "_nnPInds_DTW_rand10_single_" + s_seedId + ".txt";
		exportMatrix(nnPIndsAll_DTW, fName, v_intWarps.size(), numTrain - numPLabeled);
		fName = outputPath + "/" + datasetName + "_nnPInds_DTWD_rand10_single_" + s_seedId + ".txt";
		exportMatrix(nnPIndsAll_DTWD, fName, v_intWarps.size(), numTrain - numPLabeled);
	}

	free(trainTss);
	free(trainLabels);
	free(preLabels);
	free(seeds);
	for (int i = 0; i < numWarps; i++)
		free(allPDistMtx_DTW[i]);
	free(allPDistMtx_DTW);
	free(pDistMtx_DTWD);
	free(rankedInds);
	free(minNNDists);
	free(ranked);
	free(precisions);
	free(recalls);
	free(fscores);
	free(nnDists);
	free(nnInds);
	free(rankedInds_WK);
	free(minNNDists_WK);
	free(ranked_WK);
	free(validU_WK);
	free(precisions_WK);
	free(recalls_WK);
	free(fscores_WK);
	free(preLabels_WK);
	free(hypoSeq);
	free(nextSeq);
	free(rankedIndsAll_DTW);
	free(rankedIndsAll_DTWD);
	free(rankedIndsAll_WK_DTW);
	free(rankedIndsAll_WK_DTWD);
	free(preNumPsAll_DTW);
	free(preNumPsAll_DTWD);
	free(fscoresAll_DTW);
	free(fscoresAll_DTWD);
	free(nnPInds);
	free(nnPInds_WK);
	free(nnPIndsAll_DTW);
	free(nnPIndsAll_DTWD);
	free(nnPIndsAll_WK_DTW);
	free(nnPIndsAll_WK_DTWD);

	return 0;
}
