#include "utilities.h"
#include "calcUtilities.h"
#include "evaluationUtilities.h"
#include "STTreeNode.h"
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#define MAX_CHAR 10
#define MAX_CHAR_PER_LINE 200000

void getAllAbsoluteWarps(int* absoluteWarps, double* warps, const int numWarps, const int tsLen) {
	for (int i = 0; i < numWarps; i++) {
		absoluteWarps[i] = ceil(tsLen * warps[i]);
	}
}

void getNonDupWarps(std::vector<int>& uniqueAbsoluteWarps, std::vector<int>& nonDupWarpInds, int* absoluteWarps, const int numWarps) {

	uniqueAbsoluteWarps.clear();
	nonDupWarpInds.clear();

	int aWarp;
	for (int i = 0; i < numWarps; i++) {
		aWarp = absoluteWarps[i];
		if (!ismember(aWarp, uniqueAbsoluteWarps)) {
			uniqueAbsoluteWarps.push_back(aWarp);
			nonDupWarpInds.push_back(i);
		}
	}
}

void getValidOptions(std::vector<int>& validInds, std::vector<int>& validRows_DTW, std::vector<int>& validCols_DTW,
	std::vector<int>& validRows_DTWD, std::vector<int>& validCols_DTWD, std::vector<int> nonDupWarpInds, int numWarps) {

	validInds.clear();
	validRows_DTW.clear();
	validCols_DTW.clear();
	validRows_DTWD.clear();
	validCols_DTWD.clear();

	std::vector<int> validRows, validCols;
	int ind, offset = 0;
	for (int i = 0; i < 2; i++) {	//DTW or DTWD

		validRows.clear();
		validCols.clear();

		for (int j = 0; j < numWarps; j++) {	//warp

			if (i == 1 && j == 0)	//Discard zero warping for DTW-D
				continue;
			if (!ismember(j, nonDupWarpInds))	//Discard same absolute windows
				continue;

			for (int k = 2; k < 33; k++) {	//Discard oracle and WK stopping criteria
				ind = j * 33 + k;
				validInds.push_back(offset + ind);
				validRows.push_back(j);
				validCols.push_back(k);
			}
		}

		if (i == 0) {
			validRows_DTW = validRows;
			validCols_DTW = validCols;
		}
		else {
			validRows_DTWD = validRows;
			validCols_DTWD = validCols;
		}

		offset += numWarps * 33;
	}
}

void getRankingsByInd(int* rankingsByInd, int* curRankedInds, int numTrain) {
	for (int i = 0; i < numTrain; i++) {
		int curInd = curRankedInds[i];
		rankingsByInd[curInd] = i;
	}
}

int getPreLabel(int ind, int* rankingsByInd, int preNumP) {
	if (preNumP <= 0)
		return -1;
	else {
		if (rankingsByInd[ind] < preNumP)
			return 1;
		else
			return 0;
	}
}

void getUnweightedScore_KLD_Uncertainty(double& maxKLD, double& uncertainty,
	int& consensusLabel, double& consensusConf, double* confsByOption,
	int* preLabelsByOption, int numOptions) {
	int numPreP, numPreN;
	numPreP = numPreN = 0;
	for (int i = 0; i < numOptions; i++) {
		if (preLabelsByOption[i] == 1) {
			numPreP++;
			confsByOption[i] = 1;
		}
		else if (preLabelsByOption[i] == 0) {
			numPreN++;
			confsByOption[i] = 1;
		}
		else
			confsByOption[i] = -2;
	}

	if (numPreP == 0 || numPreN == 0) {
		if (numPreP == numPreN) {
			std::cout << "Error detected!" << std::endl;
			system("pause");
		}

		consensusConf = 1;
		if (numPreP > 0)
			consensusLabel = 1;
		else
			consensusLabel = 0;

		maxKLD = uncertainty = 0;
	}
	else {
		double consensus_prob[]{ (double)numPreP / (numPreP + numPreN), (double)numPreN / (numPreP + numPreN) };
		consensusLabel = consensus_prob[0] > consensus_prob[1] ? 1 : 0;
		consensusConf = consensus_prob[0] > consensus_prob[1] ? consensus_prob[0] : consensus_prob[1];

		uncertainty = 1 - consensusConf;

		double vote_prob[2], KLD;
		maxKLD = 0;
		for (int i = 0; i < numOptions; i++) {
			if (preLabelsByOption[i] < 0)
				continue;
			vote_prob[0] = preLabelsByOption[i];
			vote_prob[1] = 1 - vote_prob[0];
			KLD = 0;
			for (int j = 0; j < 2; j++) {
				if (doubleIsZero(vote_prob[j]))
					continue;
				KLD += vote_prob[j] * log(vote_prob[j] / consensus_prob[j]);
			}
			if (maxKLD < KLD)
				maxKLD = KLD;
		}
	}
}

void buildSTTree(STTreeNode** tree, int* nnPInds, int* rankedInds, int numTrain, int numPLabeled) {

	int ind, nnPInd;
	for (int i = 0; i < numTrain; i++) {
		ind = rankedInds[i];

		if (i < numPLabeled)
			tree[ind] = new STTreeNode(ind, -1, numTrain);
		else {
			nnPInd = nnPInds[i - numPLabeled];
			tree[ind] = new STTreeNode(ind, nnPInd, numTrain);
			tree[nnPInd]->addChild(ind);
		}
	}

}

void getInfluence(int targetInd, STTreeNode** tree, int* influences, int* preLabels, int* rankedInds) {

	if (influences[targetInd] == -1) {

		STTreeNode* node = tree[targetInd];
		int targetPreLabel = preLabels[targetInd];

		std::vector<int> childInds = node->childInds;
		if (childInds.empty())
			influences[targetInd] = 0;
		else {
			int numChildren = childInds.size();
			int influence = 0;
			for (int i = 0; i < numChildren; i++) {
				int childInd = childInds[i];
				if (preLabels[childInd] != targetPreLabel)
					continue;

				getInfluence(childInd, tree, influences, preLabels, rankedInds);
				influence += influences[childInd] + 1;
				node->updateIsInfluenced(childInd);
				node->updateIsInfluenced(tree[childInd]->influencedIndsByRanking);
			}
			influences[targetInd] = influence;

			node->setInfluencedIndsByRanking(rankedInds);
		}
	}
}

void getInfluences(int* influences, STTreeNode** tree, int* preLabels, int* rankedInds, int numTrain, double adjustFactor) {
	for (int i = 0; i < numTrain; i++) {
		influences[i] = -1;
	}

	for (int i = 0; i < numTrain; i++) {
		getInfluence(i, tree, influences, preLabels, rankedInds);
	}
}

void checkDuplicateOptions(std::vector<int>& nonDupFirstOccur, std::vector<std::vector<int>>& occursByDup, std::vector<int>& numsOccur,
	std::vector<int> validRows_DTW, std::vector<int> validRows_DTWD, int* rankedIndsAll_DTW, int* rankedIndsAll_DTWD, int* preNumPByOption, int* nnPIndsAll_DTW, int* nnPIndsAll_DTWD,
	int maxNumOptions, int maxNumOptions_DTW, int numTrain, int numPLabeled) {

	nonDupFirstOccur.clear();
	occursByDup.clear();
	numsOccur.clear();
	int offset, * rankedInds_1, * rankedInds_2, preNumP_1, preNumP_2, * nnPInds_1, * nnPInds_2, * rankedIndsAll, * nnPIndsAll;
	bool flg;
	std::vector<int*> nonDupRankedIndsAll, nonDupNNPIndsAll;
	std::vector<int> validRows, nonDupPreNumPsAll;
	for (int i = 0; i < maxNumOptions; i++) {
		if (i < maxNumOptions_DTW) {
			offset = 0;
			rankedIndsAll = rankedIndsAll_DTW;
			nnPIndsAll = nnPIndsAll_DTW;
			validRows = validRows_DTW;
		}
		else {
			offset = maxNumOptions_DTW;
			rankedIndsAll = rankedIndsAll_DTWD;
			nnPIndsAll = nnPIndsAll_DTWD;
			validRows = validRows_DTWD;
		}

		preNumP_1 = preNumPByOption[i];
		if (preNumP_1 < 0) {
			continue;
		}
		rankedInds_1 = rankedIndsAll + validRows[i - offset] * numTrain;
		nnPInds_1 = nnPIndsAll + validRows[i - offset] * (numTrain - numPLabeled);

		flg = 0;
		for (int j = 0; j < nonDupFirstOccur.size(); j++) {
			preNumP_2 = nonDupPreNumPsAll[j];
			rankedInds_2 = nonDupRankedIndsAll[j];
			nnPInds_2 = nonDupNNPIndsAll[j];
			if (isequal(rankedInds_1, rankedInds_2, numTrain) &&
				isequal(nnPInds_1, nnPInds_2, numTrain - numPLabeled) &&
				preNumP_1 == preNumP_2) {
				occursByDup[j].push_back(i);
				numsOccur[j]++;
				flg = 1;
				break;
			}

		}
		if (!flg) {
			nonDupFirstOccur.push_back(i);
			std::vector<int> tmp{ i };
			occursByDup.push_back(tmp);
			numsOccur.push_back(1);
			nonDupPreNumPsAll.push_back(preNumP_1);
			nonDupRankedIndsAll.push_back(rankedInds_1);
			nonDupNNPIndsAll.push_back(nnPInds_1);
		}
	}

}

int main(int argc, char** argv) {

	if (argc < 5) {
		printf("Exiting due to too few input arguments.\n");
		exit(1);
	}
	const std::string datasetName = argv[1];
	const int numTrain = atoi(argv[2]);
	const int numP = atoi(argv[3]);
	const int tsLen = atoi(argv[4]);
	const int numRandRuns = argc > 5 ? atoi(argv[5]) : 10;	//Do NOT change this setting!
	const int minIntWarp = argc > 6 ? atoi(argv[6]) : 0;	//Do NOT change this setting!
	const int maxIntWarp = argc > 7 ? atoi(argv[7]) : 20;	//Do NOT change this setting!
	const int intWarpStep = argc > 8 ? atoi(argv[8]) : 1;	//Do NOT change this setting!
	const std::string datasetPath = argc > 9 ? argv[9] : "../Data";
	const std::string ST1NNPath = argc > 10 ? argv[10] : "../Results/ST1NN";
	const std::string seedPath = argc > 11 ? argv[11] : "../Seeds/ModelSelection";
	const std::string outputPath = argc > 12 ? argv[12] : "../Results/ModelSelection";

	const int maxNumLabeled = numTrain;
	const int numSeeds = min(numP, 10);
	const int initNumPLabeled = 1;
	const double adjustFactor = 1;	//Legacy of a previous version. Just keep it this way.
	const double pAdjustTh = 1;	//Legacy of a previous version. Just keep it this way.

	const std::vector<int> v_wDisagree{ 0 };
	const std::vector<int> v_scoreMethod{ 0, 1, 2, 3, 4, 5 };
	const std::vector<int> v_evalMethod{ 0, 1, 2 };
	const std::vector<std::string> v_str_wDisagree{ "uwDisagree" };
	const std::vector<std::string> v_str_scoreMethod{ "Un", "Dis", "Inf", "InfUn", "InfKLD", "Rand" };
	const std::vector<std::string> v_str_evalMethod = { "NV", "NFE", "IFE" };
	int numEvalMethods = v_evalMethod.size();
	
	if (access(outputPath.c_str(), F_OK) == -1)
		mkdir(outputPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	//load time series
	size_t trainTssBytes = numTrain * tsLen * sizeof(double);
	double* trainTss = (double*)malloc(trainTssBytes);
	size_t trainLabelsBytes = numTrain * sizeof(int);
	int* trainLabels = (int*)malloc(trainLabelsBytes);
	importTimeSeries(trainTss, trainLabels, datasetPath, datasetName, "TRAIN", numTrain, tsLen);
	relabel(trainLabels, numTrain, 1);

	//DTW warping window
	std::vector<int> v_intWarps;
	int intWarp;
	for (intWarp = minIntWarp; intWarp <= maxIntWarp; intWarp += intWarpStep) {
		v_intWarps.push_back(intWarp);
	}
	const int numWarps = v_intWarps.size();

	//obtain absolute warps
	double* warps = (double*)malloc(numWarps * sizeof(double));
	for (int i = 0; i < numWarps; i++) {
		warps[i] = (double)v_intWarps[i] / 100;
	}

	int* absoluteWarps = (int*)malloc(numWarps * sizeof(int));
	getAllAbsoluteWarps(absoluteWarps, warps, numWarps, tsLen);

	//get all valid options
	std::vector<int> uniqueAbsoluteWarps, nonDupWarpInds;
	getNonDupWarps(uniqueAbsoluteWarps, nonDupWarpInds, absoluteWarps, numWarps);
	std::vector<int> validInds, validRows_DTW, validCols_DTW, validRows_DTWD, validCols_DTWD;
	getValidOptions(validInds, validRows_DTW, validCols_DTW, validRows_DTWD, validCols_DTWD, nonDupWarpInds, numWarps);
	int maxNumOptions = validInds.size();
	int maxNumOptions_DTW = validRows_DTW.size();
	int maxNumOptions_DTWD = validRows_DTWD.size();

	int numRounds = maxNumLabeled - initNumPLabeled;

	int* nextIndsToLabel_randSamp = safeMalloc<int>(numRandRuns * numRounds);
	char s_numRandRuns[MAX_CHAR];
	sprintf(s_numRandRuns, "%d", numRandRuns);

	int* rankedIndsAll_DTW = (int*)malloc(numWarps * numTrain * sizeof(int));
	int* rankedIndsAll_DTWD = (int*)malloc(numWarps * numTrain * sizeof(int));
	int* preNumPsAll_DTW = (int*)malloc(numWarps * 33 * sizeof(int));
	int* preNumPsAll_DTWD = (int*)malloc(numWarps * 33 * sizeof(int));
	double* fscoresAll_DTW = (double*)malloc(numWarps * 33 * sizeof(double));
	double* fscoresAll_DTWD = (double*)malloc(numWarps * 33 * sizeof(double));
	double* accAll_DTW = safeMalloc<double>(maxNumOptions_DTW);
	double* accAll_DTWD = safeMalloc<double>(maxNumOptions_DTWD);
	int* rankingsByIndAll_DTW = safeMalloc<int>(numWarps * numTrain);
	int* rankingsByIndAll_DTWD = safeMalloc<int>(numWarps * numTrain);
	int* preLabelsAll_DTW = safeMalloc<int>(maxNumOptions_DTW * numTrain);
	int* preLabelsAll_DTWD = safeMalloc<int>(maxNumOptions_DTWD * numTrain);
	int* preLabelsByOptionAll = safeMalloc<int>(numTrain * maxNumOptions);
	int* rankingsByOptionAll = safeMalloc<int>(numTrain * maxNumOptions);
	int* preNumPsByOption = safeMalloc<int>(maxNumOptions);
	int* nnPIndsAll_DTW = (int*)malloc(numWarps * (numTrain - initNumPLabeled) * sizeof(int));
	int* nnPIndsAll_DTWD = (int*)malloc(numWarps * (numTrain - initNumPLabeled) * sizeof(int));
	STTreeNode** tree = safeMalloc<STTreeNode*>(numTrain);
	int* consensusLabels = safeMalloc<int>(numTrain);
	double* consensusConfs = safeMalloc<double>(numTrain);
	double* confsByOptionAll = safeMalloc<double>(numTrain * maxNumOptions);
	double* KLDScores = safeMalloc<double>(numTrain);
	double* uncertaintyScores = safeMalloc<double>(numTrain);
	double* totalInfluences = safeMalloc<double>(numTrain);
	int* influencesAll = safeMalloc<int>(maxNumOptions * numTrain);
	double* influenceWeightsAll = safeMalloc<double>(maxNumOptions * numTrain);
	bool* belowPAdjustThAll = safeMalloc<bool>(maxNumOptions * numTrain);
	double* scores = safeMalloc<double>(numTrain);
	bool* isActivelyLabeled = safeMalloc<bool>(numTrain);
	int* queriedIndsByRound = safeMalloc<int>(numRounds);

	std::vector<int> validRows, validCols, nonDupFirstOccur, numsOccur;
	std::vector<std::vector<int>> occursByDup;

	char s_seedId[MAX_CHAR];
	int numLabeled, curMaxNumOptions, curPreNumP, offset, otherOffset, nextIndToLabel, nextLabel, maxVotes, numWinners, winnerInd, idx, numNonZeroDisagreement, tmpIdx, estNumP, consLabel, maxInfluence, minInfluence, curOption, curOption_1, adjNumInfluenced, label, numGapsBetweenQueries, nextCheckIdx, ind,
		* rankedIndsAll, * curRankedInds, * preNumPsAll, * rankingsByIndAll, * curRankingsByInd, * preLabelsAll, * nnPIndsAll, * curNNPInds, * preLabelsByOption, * rankingsByOption, * curWinnerInds, * curWinnerIdxes, * curPreLabels, * STTreeMatrix, * line, preLabel, * influences, * corrPreLabels, * nonDupPseudoLabels, * nonDupEstimatedTps;
	double maxScore, * fscoresAll, * curWinnerFscores, * accAll, fscore, acc, maxEstFscore, * confsByOption, * influenceWeights, conf, confP, confN, cur, factor, p, r, avgTimeBetweenQueries;
	bool* belowPAdjustTh;
	clock_t tic, toc;

	double** avgFscoreByRound_ALL = safeMalloc<double*>(numEvalMethods);
	double** avgAccByRound_ALL = safeMalloc<double*>(numEvalMethods);
	for (int i = 0; i < numEvalMethods; i++) {
		avgFscoreByRound_ALL[i] = safeMalloc<double>(numRounds);
		avgAccByRound_ALL[i] = safeMalloc<double>(numRounds);
	}
	int* nonDupNumWinners_ALL = safeMalloc<int>(numEvalMethods);

	std::vector<int> influencedInds, adjInfIndsWithRoot;
	std::string fName;
	for (int seedId = 0; seedId < numSeeds; seedId++) {
		sprintf(s_seedId, "%d", seedId);
		std::cout << "Seed = " << seedId << std::endl;

		numGapsBetweenQueries = 0;
		avgTimeBetweenQueries = 0;

		//load rankedIndsAll
		fName = ST1NNPath + "/" + datasetName + "_rankedInds_DTW_rand10_single_" + s_seedId + ".txt";
		importMatrix(rankedIndsAll_DTW, fName, numWarps, numTrain, 1);
		fName = ST1NNPath + "/" + datasetName + "_rankedInds_DTWD_rand10_single_" + s_seedId + ".txt";
		importMatrix(rankedIndsAll_DTWD, fName, numWarps, numTrain, 1);

		//load preNumPsAll
		fName = ST1NNPath + "/" + datasetName + "_preNumPs_DTW_rand10_single_" + s_seedId + ".txt";
		importMatrix(preNumPsAll_DTW, fName, numWarps, 33, 1);
		fName = ST1NNPath + "/" + datasetName + "_preNumPs_DTWD_rand10_single_" + s_seedId + ".txt";
		importMatrix(preNumPsAll_DTWD, fName, numWarps, 33, 1);

		//load fscores
		fName = ST1NNPath + "/" + datasetName + "_transFscores_DTW_rand10_single_" + s_seedId + ".txt";
		importMatrix(fscoresAll_DTW, fName, numWarps, 33, 0);
		fName = ST1NNPath + "/" + datasetName + "_transFscores_DTWD_rand10_single_" + s_seedId + ".txt";
		importMatrix(fscoresAll_DTWD, fName, numWarps, 33, 0);

		//load nnPInds
		fName = ST1NNPath + "/" + datasetName + "_nnPInds_DTW_rand10_single_" + s_seedId + ".txt";
		importMatrix(nnPIndsAll_DTW, fName, numWarps, numTrain - initNumPLabeled, 1);
		fName = ST1NNPath + "/" + datasetName + "_nnPInds_DTWD_rand10_single_" + s_seedId + ".txt";
		importMatrix(nnPIndsAll_DTWD, fName, numWarps, numTrain - initNumPLabeled, 1);

		//load sampling orders for Rand
		fName = seedPath + "/" + datasetName + "_randSampSequences_rand10_single_" + s_numRandRuns + "_" + s_seedId + ".txt";
		importMatrix(nextIndsToLabel_randSamp, fName, numRandRuns, numRounds, 1);
		for (int i = 0; i < numRandRuns * numRounds; i++) {
			nextIndsToLabel_randSamp[i]--;	//matlab -> c
		}

		for (int i = 0; i < 2; i++) {
			rankedIndsAll = i == 0 ? rankedIndsAll_DTW : rankedIndsAll_DTWD;
			rankingsByIndAll = i == 0 ? rankingsByIndAll_DTW : rankingsByIndAll_DTWD;
			preNumPsAll = i == 0 ? preNumPsAll_DTW : preNumPsAll_DTWD;
			preLabelsAll = i == 0 ? preLabelsAll_DTW : preLabelsAll_DTWD;
			validRows = i == 0 ? validRows_DTW : validRows_DTWD;
			validCols = i == 0 ? validCols_DTW : validCols_DTWD;
			curMaxNumOptions = i == 0 ? maxNumOptions_DTW : maxNumOptions_DTWD;

			//obtain rankingsByInd
			for (int j = 0; j < numWarps; j++) {
				curRankedInds = rankedIndsAll + j * numTrain;
				curRankingsByInd = rankingsByIndAll + j * numTrain;
				getRankingsByInd(curRankingsByInd, curRankedInds, numTrain);
			}

			//get predicted labels
			for (int j = 0; j < curMaxNumOptions; j++) {
				curRankingsByInd = rankingsByIndAll + validRows[j] * numTrain;
				curPreNumP = preNumPsAll[validRows[j] * 33 + validCols[j]];
				for (int k = 0; k < numTrain; k++) {
					preLabelsAll[j * numTrain + k] = getPreLabel(k, curRankingsByInd, curPreNumP);
				}
			}

			accAll = i == 0 ? accAll_DTW : accAll_DTWD;
			for (int j = 0; j < curMaxNumOptions; j++) {
				curPreLabels = preLabelsAll + j * numTrain;
				accAll[j] = accWithSeed(trainLabels, curPreLabels, rankedIndsAll_DTW, numTrain, initNumPLabeled);
			}
		}

		//get preLabelsByOption
		for (int i = 0; i < numTrain; i++) {
			preLabelsByOption = preLabelsByOptionAll + i * maxNumOptions;
			for (int j = 0; j < maxNumOptions; j++) {
				preLabelsAll = j < maxNumOptions_DTW ? preLabelsAll_DTW : preLabelsAll_DTWD;
				offset = j < maxNumOptions_DTW ? 0 : maxNumOptions_DTW;
				preLabelsByOption[j] = preLabelsAll[(j - offset) * numTrain + i];
			}
		}

		//get preNumPsByOption
		for (int j = 0; j < maxNumOptions; j++) {
			preNumPsAll = j < maxNumOptions_DTW ? preNumPsAll_DTW : preNumPsAll_DTWD;
			validRows = j < maxNumOptions_DTW ? validRows_DTW : validRows_DTWD;
			validCols = j < maxNumOptions_DTW ? validCols_DTW : validCols_DTWD;
			offset = j < maxNumOptions_DTW ? 0 : maxNumOptions_DTW;
			preNumPsByOption[j] = preNumPsAll[validRows[j - offset] * 33 + validCols[j - offset]];
		}

		//get rankingsByOption
		for (int i = 0; i < numTrain; i++) {
			rankingsByOption = rankingsByOptionAll + i * maxNumOptions;
			for (int j = 0; j < maxNumOptions; j++) {
				rankingsByIndAll = j < maxNumOptions_DTW ? rankingsByIndAll_DTW : rankingsByIndAll_DTWD;
				validRows = j < maxNumOptions_DTW ? validRows_DTW : validRows_DTWD;
				offset = j < maxNumOptions_DTW ? 0 : maxNumOptions_DTW;
				rankingsByOption[j] = rankingsByIndAll[validRows[j - offset] * numTrain + i];
			}
		}

		checkDuplicateOptions(nonDupFirstOccur, occursByDup, numsOccur, validRows_DTW, validRows_DTWD, rankedIndsAll_DTW,
			rankedIndsAll_DTWD, preNumPsByOption, nnPIndsAll_DTW, nnPIndsAll_DTWD, maxNumOptions, maxNumOptions_DTW, numTrain, initNumPLabeled);

		int numNonDupOptions = nonDupFirstOccur.size();
		STTreeNode** nonDupSTTreesAll = safeMalloc<STTreeNode*>(numNonDupOptions * numTrain);
		memset(belowPAdjustThAll, 0, maxNumOptions * numTrain * sizeof(bool));
		int cnt = 0;
		int nextFirstOccur = nonDupFirstOccur[0];
		for (int i = 0; i < maxNumOptions; i++) {

			belowPAdjustTh = belowPAdjustThAll + i * numTrain;
			influences = influencesAll + i * numTrain;
			if (preNumPsByOption[i] < 0) {
				for (int j = 0; j < numTrain; j++) {
					influences[j] = -2;
				}
				continue;
			}

			//set curPreLabels
			preLabelsAll = i < maxNumOptions_DTW ? preLabelsAll_DTW : preLabelsAll_DTWD;
			offset = i < maxNumOptions_DTW ? 0 : maxNumOptions_DTW;
			curPreLabels = preLabelsAll + (i - offset) * numTrain;

			//set curRankedInds
			validRows = i < maxNumOptions_DTW ? validRows_DTW : validRows_DTWD;
			rankedIndsAll = i < maxNumOptions_DTW ? rankedIndsAll_DTW : rankedIndsAll_DTWD;
			curRankedInds = rankedIndsAll + validRows[i - offset] * numTrain;

			//set curNNPInds
			nnPIndsAll = i < maxNumOptions_DTW ? nnPIndsAll_DTW : nnPIndsAll_DTWD;
			curNNPInds = nnPIndsAll + validRows[i - offset] * (numTrain - initNumPLabeled);;

			//build STTree
			buildSTTree(tree, curNNPInds, curRankedInds, numTrain, initNumPLabeled);

			getInfluences(influences, tree, curPreLabels, curRankedInds, numTrain, adjustFactor);
			maxInfluence = max(influences, numTrain);
			for (int j = 0; j < numTrain; j++) {
				cur = (double)influences[j] / maxInfluence;
				if (!doubleGeq(cur, pAdjustTh))
					belowPAdjustTh[j] = 1;
			}

			if (i == nextFirstOccur) {
				for (int j = 0; j < numTrain; j++) {
					nonDupSTTreesAll[cnt * numTrain + j] = new STTreeNode();
					tree[j]->copyTo(nonDupSTTreesAll[cnt * numTrain + j]);
				}

				cnt++;
				nextFirstOccur = cnt < numNonDupOptions ? nonDupFirstOccur[cnt] : -1;
			}

			for (int j = 0; j < numTrain; j++) {
				delete tree[j];
			}
		}

		int* nonDupVotesByOption = safeMalloc<int>(numNonDupOptions);
		int** nonDupPseudoLabelsAll_ALL = safeMalloc<int*>(numEvalMethods);
		int** nonDupEstNumPsInU_ALL = safeMalloc<int*>(numEvalMethods);
		int** nonDupEstimatedTpsAll_ALL = safeMalloc<int*>(numEvalMethods);
		double** nonDupEstTotalFscores_ALL = safeMalloc<double*>(numEvalMethods);
		int** nonDupMaxIdxes_ALL = safeMalloc<int*>(numEvalMethods);
		int** numOccur_maxIdxes_ALL = safeMalloc<int*>(numEvalMethods);
		for (int i = 0; i < numEvalMethods; i++) {
			nonDupPseudoLabelsAll_ALL[i] = safeMalloc<int>(numNonDupOptions * numTrain);
			nonDupEstNumPsInU_ALL[i] = safeMalloc<int>(numNonDupOptions);
			nonDupEstimatedTpsAll_ALL[i] = safeMalloc<int>(numNonDupOptions * numNonDupOptions);
			nonDupEstTotalFscores_ALL[i] = safeMalloc<double>(numNonDupOptions);
			nonDupMaxIdxes_ALL[i] = safeMalloc<int>(numNonDupOptions);
			numOccur_maxIdxes_ALL[i] = safeMalloc<int>(numNonDupOptions);
		}

		for (int wDisagreeId = 0; wDisagreeId < v_wDisagree.size(); wDisagreeId++) {	//Legacy of a previous version. Useless now.

			int wDisagree = v_wDisagree[wDisagreeId];
			std::string s_wDisagree = v_str_wDisagree[wDisagreeId];

			//get KLD and uncertainty scores
			for (int i = 0; i < numTrain; i++) {
				preLabelsByOption = preLabelsByOptionAll + i * maxNumOptions;
				rankingsByOption = rankingsByOptionAll + i * maxNumOptions;
				confsByOption = confsByOptionAll + i * maxNumOptions;
				getUnweightedScore_KLD_Uncertainty(KLDScores[i], uncertaintyScores[i], consensusLabels[i], consensusConfs[i], confsByOption,
					preLabelsByOption, maxNumOptions);
			}

			for (int i = 0; i < maxNumOptions; i++) {
				influenceWeights = influenceWeightsAll + i * numTrain;
				belowPAdjustTh = belowPAdjustThAll + i * numTrain;

				if (preNumPsByOption[i] < 0) {
					for (int j = 0; j < numTrain; j++)
						influenceWeights[j] = -2;
					continue;
				}

				for (int j = 0; j < numTrain; j++) {
					if (belowPAdjustTh[j])
						influenceWeights[j] = 1;
					else {
						preLabel = preLabelsByOptionAll[j * maxNumOptions + i];
						conf = confsByOptionAll[j * maxNumOptions + i];
						confP = preLabel == 1 ? conf : 1 - conf;
						influenceWeights[j] = confP * adjustFactor + (1 - confP);
					}
				}
			}

			//total influences
			memset(totalInfluences, 0, numTrain * sizeof(double));
			for (int i = 0; i < maxNumOptions; i++) {
				if (preNumPsByOption[i] < 0)
					continue;

				influences = influencesAll + i * numTrain;
				influenceWeights = influenceWeightsAll + i * numTrain;
				for (int j = 0; j < numTrain; j++) {
					totalInfluences[j] += influences[j] * influenceWeights[j];
				}
			}

			for (int scoreMethodId = 0; scoreMethodId < v_scoreMethod.size(); scoreMethodId++) {

				int scoreMethod = v_scoreMethod[scoreMethodId];
				std::string s_score = v_str_scoreMethod[scoreMethodId];
				int numRuns = scoreMethod == 5 ? numRandRuns : 1;

				if (scoreMethod == 0) {
					memcpy(scores, uncertaintyScores, numTrain * sizeof(double));
				}
				else if (scoreMethod == 1) {
					memcpy(scores, KLDScores, numTrain * sizeof(double));
				}
				else if (scoreMethod == 2) {
					memcpy(scores, totalInfluences, numTrain * sizeof(double));
				}
				else if (scoreMethod == 3) {
					for (int i = 0; i < numTrain; i++) {
						scores[i] = totalInfluences[i] * uncertaintyScores[i];
					}
				}
				else if (scoreMethod == 4) {
					for (int i = 0; i < numTrain; i++) {
						scores[i] = totalInfluences[i] * KLDScores[i];
					}
				}
				else {
					//nothing to do for Rand
				}

				for (int i = 0; i < numEvalMethods; i++) {
					memset(avgFscoreByRound_ALL[i], 0, numRounds * sizeof(double));
					memset(avgAccByRound_ALL[i], 0, numRounds * sizeof(double));
				}

				for (int runId = 0; runId < numRuns; runId++) {

					char s_runId[MAX_CHAR];
					sprintf(s_runId, "%d", runId);

					//initiate pseudo-labels
					for (int i = 0; i < numNonDupOptions; i++) {
						curOption = nonDupFirstOccur[i];
						nonDupPseudoLabels = nonDupPseudoLabelsAll_ALL[0] + i * numTrain;

						if (curOption < maxNumOptions_DTW) {
							preLabelsAll = preLabelsAll_DTW;
							offset = 0;
						}
						else {
							preLabelsAll = preLabelsAll_DTWD;
							offset = maxNumOptions_DTW;
						}
						memcpy(nonDupPseudoLabels, preLabelsAll + (curOption - offset) * numTrain, numTrain * sizeof(int));
						nonDupEstNumPsInU_ALL[0][i] = preNumPsByOption[curOption] - initNumPLabeled;


						//Legacy of a previous version. Useless now.
						//for (int j = 0; j < numTrain; j++) {
						//	if (consensusConfs[j] > confsByOptionAll[j * maxNumOptions + curOption]) {
						//		nonDupEstNumPsInU_ALL[0][i] += consensusLabels[j] - nonDupPseudoLabels[j];
						//		nonDupPseudoLabels[j] = consensusLabels[j];
						//	}
						//}
					}
					for (int i = 1; i < numEvalMethods; i++) {
						memcpy(nonDupPseudoLabelsAll_ALL[i], nonDupPseudoLabelsAll_ALL[0], numNonDupOptions * numTrain * sizeof(int));
						memcpy(nonDupEstNumPsInU_ALL[i], nonDupEstNumPsInU_ALL[0], numNonDupOptions * sizeof(int));
					}

					memset(nonDupEstimatedTpsAll_ALL[0], 0, numNonDupOptions * numNonDupOptions * sizeof(int));
					for (int i = 0; i < numNonDupOptions; i++) {
						nonDupEstimatedTps = nonDupEstimatedTpsAll_ALL[0] + i * numNonDupOptions;
						nonDupPseudoLabels = nonDupPseudoLabelsAll_ALL[0] + i * numTrain;
						for (int j = 0; j < numNonDupOptions; j++) {

							curOption = nonDupFirstOccur[j];
							curPreNumP = preNumPsByOption[curOption];
							if (curOption < maxNumOptions_DTW) {
								rankedIndsAll = rankedIndsAll_DTW;
								validRows = validRows_DTW;
								offset = 0;
							}
							else {
								rankedIndsAll = rankedIndsAll_DTWD;
								validRows = validRows_DTWD;
								offset = maxNumOptions_DTW;
							}
							curRankedInds = rankedIndsAll + validRows[curOption - offset] * numTrain;

							for (int k = initNumPLabeled; k < curPreNumP; k++) {
								ind = curRankedInds[k];
								if (nonDupPseudoLabels[ind] == 1) {
									nonDupEstimatedTps[j]++;
								}
							}
						}
					}
					for (int i = 1; i < numEvalMethods; i++) {
						memcpy(nonDupEstimatedTpsAll_ALL[i], nonDupEstimatedTpsAll_ALL[0], numNonDupOptions * numNonDupOptions * sizeof(int));
					}

					//active learning
					memset(isActivelyLabeled, 0, numTrain * sizeof(bool));
					for (int i = 0; i < initNumPLabeled; i++) {
						isActivelyLabeled[rankedIndsAll_DTW[i]] = 1;
					}
					memset(nonDupVotesByOption, 0, numNonDupOptions * sizeof(int));
					numLabeled = initNumPLabeled;

					while (numLabeled < maxNumLabeled) {

						//query
						if (scoreMethod != 5) {
							//find the current UNLABELED example with the highest score
							maxScore = -INF;
							for (int i = 0; i < numTrain; i++) {
								if (!isActivelyLabeled[i] && scores[i] > maxScore) {
									maxScore = scores[i];
									nextIndToLabel = i;
								}
							}
						}
						else {
							nextIndToLabel = nextIndsToLabel_randSamp[runId * numRounds + (numLabeled - initNumPLabeled)];
						}

						toc = clock();
						if (numLabeled != initNumPLabeled) {
							numGapsBetweenQueries++;
							avgTimeBetweenQueries += (double)(toc - tic) / ((double)CLOCKS_PER_SEC);
						}
						tic = clock();

						nextLabel = trainLabels[nextIndToLabel];
						numLabeled++;
						isActivelyLabeled[nextIndToLabel] = 1;
						queriedIndsByRound[numLabeled - initNumPLabeled - 1] = nextIndToLabel;

						for (int i = 0; i < numNonDupOptions; i++) {
							curOption = nonDupFirstOccur[i];

							if (curOption < maxNumOptions_DTW) {
								preLabelsAll = preLabelsAll_DTW;
								offset = 0;
							}
							else {
								preLabelsAll = preLabelsAll_DTWD;
								offset = maxNumOptions_DTW;
							}
							curPreLabels = preLabelsAll + (curOption - offset) * numTrain;

							if (curPreLabels[nextIndToLabel] == nextLabel)
								nonDupVotesByOption[i]++;
						}
						maxWithTies(maxVotes, nonDupMaxIdxes_ALL[0], nonDupNumWinners_ALL[0], nonDupVotesByOption, numNonDupOptions);

						for (int i = 0; i < numEvalMethods; i++)
							memset(nonDupEstTotalFscores_ALL[i], 0, numNonDupOptions * sizeof(double));

						for (int i = 0; i < numNonDupOptions; i++) {
							curOption = nonDupFirstOccur[i];

							adjInfIndsWithRoot.clear();
							adjInfIndsWithRoot.push_back(nextIndToLabel);
							influencedInds = nonDupSTTreesAll[i * numTrain + nextIndToLabel]->influencedIndsByRanking;
							factor = nextLabel == 1 ? adjustFactor : 1;
							adjNumInfluenced = round(influencedInds.size() * factor);
							adjInfIndsWithRoot.insert(adjInfIndsWithRoot.end(), influencedInds.begin(), influencedInds.begin() + adjNumInfluenced);

							for (int j = 0; j < adjInfIndsWithRoot.size(); j++) {
								ind = adjInfIndsWithRoot[j];
								if (isActivelyLabeled[ind])
									label = trainLabels[ind];
								else
									label = nextLabel;

								if (ind == nextIndToLabel)
									nonDupEstNumPsInU_ALL[1][i] += label - nonDupPseudoLabelsAll_ALL[1][i * numTrain + ind];
								nonDupEstNumPsInU_ALL[2][i] += label - nonDupPseudoLabelsAll_ALL[2][i * numTrain + ind];

								for (int k = 0; k < numNonDupOptions; k++) {

									curOption_1 = nonDupFirstOccur[k];
									if (curOption_1 < maxNumOptions_DTW) {
										preLabelsAll = preLabelsAll_DTW;
										offset = 0;
									}
									else {
										preLabelsAll = preLabelsAll_DTWD;
										offset = maxNumOptions_DTW;
									}
									curPreLabels = preLabelsAll + (curOption_1 - offset) * numTrain;

									if (curPreLabels[ind] == 1) {

										if (ind == nextIndToLabel)
											nonDupEstimatedTpsAll_ALL[1][i * numNonDupOptions + k] += label - nonDupPseudoLabelsAll_ALL[1][i * numTrain + ind];
										nonDupEstimatedTpsAll_ALL[2][i * numNonDupOptions + k] += label - nonDupPseudoLabelsAll_ALL[2][i * numTrain + ind];
									}
								}

								if (ind == nextIndToLabel)
									nonDupPseudoLabelsAll_ALL[1][i * numTrain + ind] = label;
								nonDupPseudoLabelsAll_ALL[2][i * numTrain + ind] = label;
							}

							for (int j = 0; j < numNonDupOptions; j++) {

								curOption_1 = nonDupFirstOccur[j];
								for (int evalMethod = 1; evalMethod < numEvalMethods; evalMethod++) {
									nonDupEstimatedTps = nonDupEstimatedTpsAll_ALL[evalMethod] + i * numNonDupOptions;
									nonDupEstTotalFscores_ALL[evalMethod][j] += 2.0 * nonDupEstimatedTps[j] /
										(preNumPsByOption[curOption_1] - initNumPLabeled + nonDupEstNumPsInU_ALL[evalMethod][i]);
								}
							}
						}
						for (int evalMethod = 1; evalMethod < numEvalMethods; evalMethod++) {
							maxWithTies(maxEstFscore, nonDupMaxIdxes_ALL[evalMethod], nonDupNumWinners_ALL[evalMethod], nonDupEstTotalFscores_ALL[evalMethod], numNonDupOptions);
						}

						for (int evalMethod = 0; evalMethod < numEvalMethods; evalMethod++) {
							int* nonDupMaxIdxes = nonDupMaxIdxes_ALL[evalMethod];
							int nonDupNumWinners = nonDupNumWinners_ALL[evalMethod];
							fscore = acc = 0;
							int numOccurTotal = 0;
							for (int i = 0; i < nonDupNumWinners; i++) {
								idx = nonDupMaxIdxes[i];
								int numOccur = numsOccur[idx];
								numOccurTotal += numOccur;

								winnerInd = validInds[nonDupFirstOccur[idx]];
								if (nonDupFirstOccur[idx] < maxNumOptions_DTW) {
									fscoresAll = fscoresAll_DTW;
									accAll = accAll_DTW;
									tmpIdx = nonDupFirstOccur[idx];

								}
								else {
									fscoresAll = fscoresAll_DTWD;
									winnerInd -= numWarps * 33;
									accAll = accAll_DTWD;
									tmpIdx = nonDupFirstOccur[idx] - maxNumOptions_DTW;
								}

								if (fscoresAll[winnerInd] == -2) {
									std::cout << "Invalid fscore detected!" << std::endl;
									system("pause");
								}
								if (doubleGeq(fscoresAll[winnerInd], 0))
									fscore += fscoresAll[winnerInd] * numOccur;
								if (doubleGeq(accAll[tmpIdx], 0))
									acc += accAll[tmpIdx] * numOccur;
							}

							fscore = fscore / numOccurTotal;
							avgFscoreByRound_ALL[evalMethod][numLabeled - initNumPLabeled - 1] += fscore;
							acc = acc / numOccurTotal;
							avgAccByRound_ALL[evalMethod][numLabeled - initNumPLabeled - 1] += acc;
						}
					}
				}

				for (int evalMethod = 0; evalMethod < numEvalMethods; evalMethod++) {
					const std::string s_eval = v_str_evalMethod[evalMethod];

					for (int i = 0; i < numRounds; i++) {
						avgFscoreByRound_ALL[evalMethod][i] /= numRuns;
						avgAccByRound_ALL[evalMethod][i] /= numRuns;
					}
					fName = outputPath + "/" + datasetName + "_fscoreByRound_rand10_" + s_score + "_" + s_eval + "_" + s_seedId + ".txt";
					exportMatrix(avgFscoreByRound_ALL[evalMethod], fName, 1, numRounds, 15);
				}
			}
		}

		avgTimeBetweenQueries /= numGapsBetweenQueries;
		fName = outputPath + "/" + datasetName + "_avgTimeBetweenQueries_rand10_" + s_seedId + ".txt";
		exportMatrix(&avgTimeBetweenQueries, fName, 1, 1, 15);

		for (int i = 0; i < numNonDupOptions * numTrain; i++) {
			delete nonDupSTTreesAll[i];
		}
		free(nonDupSTTreesAll);
		free(nonDupVotesByOption);
		for (int i = 0; i < numEvalMethods; i++) {
			free(nonDupPseudoLabelsAll_ALL[i]);
			free(nonDupEstNumPsInU_ALL[i]);
			free(nonDupEstimatedTpsAll_ALL[i]);
			free(nonDupEstTotalFscores_ALL[i]);
			free(nonDupMaxIdxes_ALL[i]);
			free(numOccur_maxIdxes_ALL[i]);
		}
		free(nonDupPseudoLabelsAll_ALL);
		free(nonDupEstNumPsInU_ALL);
		free(nonDupEstimatedTpsAll_ALL);
		free(nonDupEstTotalFscores_ALL);
		free(nonDupMaxIdxes_ALL);
		free(numOccur_maxIdxes_ALL);
	}

	for (int i = 0; i < numEvalMethods; i++) {
		free(avgFscoreByRound_ALL[i]);
		free(avgAccByRound_ALL[i]);
	}
	free(avgFscoreByRound_ALL);
	free(avgAccByRound_ALL);
	free(trainTss);
	free(trainLabels);
	free(rankedIndsAll_DTW);
	free(rankedIndsAll_DTWD);
	free(preNumPsAll_DTW);
	free(preNumPsAll_DTWD);
	free(fscoresAll_DTW);
	free(fscoresAll_DTWD);
	free(warps);
	free(absoluteWarps);
	free(isActivelyLabeled);
	free(rankingsByIndAll_DTW);
	free(rankingsByIndAll_DTWD);
	free(preLabelsAll_DTW);
	free(preLabelsAll_DTWD);
	free(preLabelsByOptionAll);
	free(rankingsByOptionAll);
	free(preNumPsByOption);
	free(KLDScores);
	free(uncertaintyScores);
	free(accAll_DTW);
	free(accAll_DTWD);
	free(consensusLabels);
	free(totalInfluences);
	free(tree);
	free(scores);
	free(influencesAll);
	free(consensusConfs);
	free(confsByOptionAll);
	free(influenceWeightsAll);
	free(belowPAdjustThAll);
	free(queriedIndsByRound);
	free(nnPIndsAll_DTW);
	free(nnPIndsAll_DTWD);
	free(nextIndsToLabel_randSamp);
	free(nonDupNumWinners_ALL);
}
