# Overview
This repository holds the source code and raw experimental results of ICDE 2020 paper 441 "Active Model Selection for Positive Unlabeled Time Series Classification". This repository has the following four folders.

Code: all our source code. We will later show how to use it.

Data: a sample UCR dataset (GunPointMaleVersusFemale), as well as the data (201_A_pyramid) we used in the case study in arrhythmia detection. We will elaborate on the datasets later.
    
Results: This folder consists of all our raw results. There are three types of files in this folder:
- fscore_0.xx.xlsx: raw average F1-scores on the UCR datasets, which are used to plot the critical difference diagrams in our paper. The "0.xx" in the file names indicate the percentatage of queried U examples. For instance, "fscore_0.1.xlsx" consists of the raw results of all model selection methods on all datasets with 10% of the U examples queried. Note that the critical difference diagrams are actually plotted using "one minus F1-score" values so as to keep the better performing methods on the right of the diagrams. These values are equal to Van Rijsbergen's effectiveness measure with alpha = 0.5. See https://en.wikipedia.org/wiki/F1_score for more.
- fscore_MITDB.xlsx: raw average F1-scores on the 201_A_pyramid dataset (used in our case study) with the number of queried U examples ranging from 1 : |U|.
- responseTimes.xlsx: average user interaction response time of all model selection methods on UCR datasets. The average response time on 201_A_pyramid is 0.05s as we have reported in our paper, which has not been separately listed in the Results folder.

Seeds: seed files used by the code, including the initial PL sets used to run the candidate ST-1NN models, and sampling orders for model selection methods using the randomg sampling (Rand) strategy.

# How to use the code
Please take the following steps to obtain PUTSC model selection results.
1. Use GPU-DTW.cu to calculate the DTW distances.
2. Use ST1NN.cpp to run the candidate ST-1NN models
3. Use modelSelection.cpp to run the model selection methods.

Input parameters of GPU-DTW.cu:
- datasetName: name of dataset, for example "GunPointMaleVersusFemale"
- numTrain: number of examples in the dataset, for example "135"
- tsLen: time series length, for example "150"
- deviceId: the device ID of the selected NVIDIA GPU, default 0. If you have multiple NVIDIA GPUs, you can alter this value to the ID of the device you wish to run the code on.
- minIntWarp, maxIntWarp, intWarpStep: parameters related to DTW warping windows, which are minimum warping window * 100, maximum warping window * 100, and warping window step * 100, respectively. Do NOT change their default settings as it can lead to mistakes in subsequent outputs.
- maxThreadsPerBlock: maximum number of GPU threads per block, default 256. You can change this to lower values for large datasets.
- maxBlocksPerGrid: maximum number of GPU blocks per grid, default 256. You can change this to lower values for large datasets.
- datasetPath: path to the dataset folder
- dtwPath: output path of DTW distance matrices

Input parameters of ST1NN.cpp
- datasetName: name of dataset, for instance "GunPointMaleVersusFemale"
- numTrain: number of examples in the dataset, for instance "135"
- numP: number of positive examples, for instance "71". Note that we use the label "1" as the positive class.
- tsLen: time series length, for instance "150"
- minIntWarp, maxIntWarp, intWarpStep: parameters related to DTW warping windows, which are minimum warping window * 100, maximum warping window * 100, and warping window step * 100, respectively. Do NOT change their default settings as it can lead to mistakes in subsequent outputs.
- minNumIters, maxNumIters: minimum and maximum number of ST-1NN iterations allowed. We advise that you keep them as their default settings.
- datasetPath: path to the dataset folder
- dtwPath: path to DTW distance matrices
- seedPath: path to the file consisting of initial PL example indices for the multiple runs of ST-1NN. For our example datasets, we have attached the seed file we used in our experiments. You are welcome to use genST1NNSeeds.m to generate your own seeds, but keep in mind that the generated seeds can OVERWRITE the orignal seed files we have provided.
- outputPath: output path of ST-1NN results

Input parameters of modelSelection.cpp
- datasetName: name of dataset, for instance "GunPointMaleVersusFemale"
- numTrain: number of examples in the dataset, for instance "135"
- numP: number of positive examples, for instance "71". Note that we use the label "1" as the positive class.
- tsLen: time series length, for instance "150"
- numRandRuns: the number of runs for random sampling based methods. Do NOT change its default value as it can lead to mistakes in subsequent outputs.
- minIntWarp, maxIntWarp, intWarpStep: parameters related to DTW warping windows, which are minimum warping window * 100, maximum warping window * 100, and warping window step * 100, respectively. Do NOT change their default settings as it can lead to mistakes in subsequent outputs.
- datasetPath: path to the dataset folder 
- ST1NNPath: path to ST-1NN results
- seedPath: path to files containing the sampling orders for the random sampling (Rand) strategy. You are welcome to use genRandSampSeq.m to generate your own seeds, but keep in mind that the generated seeds can OVERWRITE the orignal seed files we have provided.
- outputPath: output path of final model selection results

Final output of modelSelection.cpp includes the following.
- F1-score files. The naming format is "\[dataset\]\_fscoreByRound\_rand10\_\[Sampling strategy\]\_\[Evaluation strategy\]\_\[ST-1NN seed ID\]\_.txt", for instance "GunPointMaleVersusFemale_fscoreByRound_rand10_Dis_IFE_3.txt". The ST-1NN seed ID is among 0-9 which correspond to the ten runs with different initial PL set. Each file consists of numU datapoints where numU is the number of U examples. The _i_-th data point is the F1-score of the selected model when the number of queried U example is _i_. If there are ties between multiple selected models, we use their average F1-score, as this is the expected F1-score for randomly picking a selected model with equal probability.
- User interaction response time files. The naming format is "\[dataset\]\_avgTimeBetweenQueries_rand10\_\[ST-1NN seed ID\]\_.txt", for instance "GunPointMaleVersusFemale_avgTimeBetweenQueries_rand10_3.txt". Each file consists of only one datapoint, which is the average user interaction response time (in seconds) for the current ST-1NN seed.

# On the datasets
In our paper, we have used two data sources: the UCR archive and MIT-BIH Arrhythmia Database (MITDB). Their references and web links can be found in our paper.

As with UCR datasets, we have used the original data without further editing. The complete datasets are available at http://www.cs.ucr.edu/~eamonn/time_series_data_2018/. Note that for datasets with missing values and variable time series lengths, the UCR archive has provided officially preprocessed versions of them, which is the data we have used in our experiments.

As with MITDB data, we have used lead A of Record 201. The original data is available at https://physionet.org/content/mitdb/1.0.0/. We have preprocessed the data following the practice of this paper below. 

    J. He, L. Sun, J. Rong, H. Wang, and Y. Zhang, "A pyramid-like model for heartbeat classification from ECG recordings," PLOS ONE, vol. 13, pp. 1–19, 11 2018

For the data preprocessing source code, see https://github.com/SamHO666/A-Pyramid-like-Model-for-Heartbeat-Classification. For user convenience, we have attached the preprocessed dataset (in UCR format, see 201_A_pyramid_TRAIN.tsv in the Data folder). Note that the data has been relabeled such that the VEB class is labeled as "1", while all other data is labeled as "0".

Note that for UCR datasets, we only used the training sets. This is because in transductive PUTSC, there are no testing sets in the conventional sense. Therefore, following the practice of the paper listed below, we only used the training sets in our experiments.

    M. G. Castellanos, C. Bergmeir, I. Triguero, Y. Rodríguez, J. M. Benítez, "On the stopping criteria for k-Nearest Neighbor in positive unlabeled time series classification problems," Inf. Sci. 328, pp. 42-59, 2016
    
As with MITDB data, we used the entire record (a small proportion of the data is discarded in the preprocessing phase) which was not separated into training and testing sets by the original contributors. The file "201_A_pyramid_TRAIN.tsv" in the Data folder was named this way only to facilitate file reading by our code. It does NOT correspond to an actual training set.
