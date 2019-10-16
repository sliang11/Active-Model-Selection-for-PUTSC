This repository holds the source code and raw experimental results of ICDE 2020 paper 441 "Active Model Selection for Positive Unlabeled Time Series Classification".

# Overview
This repository has the following folders.

Code: all our source code. We will later show how to use the paper.

Data: a sample UCR dataset (GunPointMaleVersusFemale), as well as the data (201_A_pyramid) we used in the case study in arrhythmia detection. The latter has been formatted in the same way as the UCR datasets. The label "1" correspond to the VEB class, while the label "0" correspond to other types of heartbeats. 

Note that we have only provided the training set. This is because in transductive PUTSC, there are no testing sets in the conventional sense. Therefore, following the practice of the paper listed below, we only used the training sets in our experiments.

    Mabel González Castellanos, Christoph Bergmeir, Isaac Triguero, Yanet Rodríguez, José Manuel Benítez: On the stopping criteria for k-Nearest Neighbor in positive unlabeled time series classification problems. Inf. Sci. 328: 42-59 (2016)
    
Results: This folder consists of all our raw results. There are three types of files in this folder:
- Files that are formatted as "fscore_0.xx.xlsx" are raw average F1-scores on the UCR datasets. The "0.xx" in the file name indicate the percentatage of queried U examples. For instance, "fscore_0.1.xlsx" consists of the raw results of all model selection methods on all the datasets with 10% of the U examples queried.
- fscore_MITDB.xlsx: raw average F1-scores on the 201_A_pyramid dataset, which we have used in our case study.
- responseTimes.xlsx: average user interaction response time of all model selection methods on all the UCR datasets we have used. The average response time on 201_A_pyramid is 0.05s as we have reported in our paper.

# How to use the code
Please take the following steps to obtain PUTSC model selection results.
1. Use GPU-DTW.cu to calculate the DTW distances.
2. Use ST1NN.cpp to run the candidate ST-1NN models
3. Use modelSelection.cpp to run the model selection methods.

Input parameters of GPU-DTW.cu:
- datasetName: name of dataset, for example "GunPointMaleVersusFemale"
- numTrain: number of examples in the dataset, for example "135"
- tsLen: time series length, for example "150"
- deviceId: GPU ID, default 0. If you have multiple GPUs, you can set it to other values to use a different GPU.
- minIntWarp, maxIntWarp, intWarpStep: optional parameters related to DTW warping windows. Do NOT change their default settings as it can lead to mistakes in subsequent outputs.
- maxThreadsPerBlock: maximum number of GPU threads per block, default 256
- maxBlocksPerGrid: maximum number of GPU blocks per grid, default 256
- datasetPath: path to the dataset folder
- dtwPath: output path of DTW distance matrices

Input parameters of ST1NN.cpp
- datasetName: name of dataset, for example "GunPointMaleVersusFemale"
- numTrain: number of examples in the dataset, for example "135"
- numP: number of positive examples, for example "71". We use the label "1" as the positive class.
- tsLen: time series length, for example "150"
- minIntWarp, maxIntWarp, intWarpStep: optional parameters related to DTW warping windows. Do NOT change their default settings as it can lead to mistakes in subsequent outputs.
- minNumIters, maxNumIters: minimum and maximum number of ST-1NN iterations allowed.
- datasetPath: path to the dataset folder
- dtwPath: path to DTW distance matrices
- seedPath: path to the file consisting of initial PL example indices for ST-1NN. You can use genST1NNSeeds.m to generate the seeds. For our example datasets, we have attached the seed file we used in our experiments.
- outputPath: output path of ST-1NN results

Input parameters of modelSelection.cpp
- datasetName: name of dataset, for example "GunPointMaleVersusFemale"
- numTrain: number of examples in the dataset, for example "135"
- numP: number of positive examples, for example "71". We use the label "1" as the positive class.
- tsLen: time series length, for example "150"
- numRandRuns: the number of runs for random sampling based methods. Do NOT change its default value as it can lead to mistakes in subsequent outputs.
- minIntWarp, maxIntWarp, intWarpStep: optional parameters related to DTW warping windows. Do NOT change their default settings as it can lead to mistakes in subsequent outputs.
- datasetPath: path to the dataset folder 
- ST1NNPath: path to ST-1NN results
- seedPath: path to files containing the random sampling orders for the random sampling strategy.
- outputPath: output path of model selection results

Final output of modelSelection.cpp
- F1-score files. The naming format is "\[dataset\]\_fscoreByRound\_rand10\_\[Sampling strategy\]\_\[ST-1NN seed ID\]\_.txt", for example "GunPointMaleVersusFemale_fscoreByRound_rand10_Dis_IFE_3.txt". The ST-1NN seed ID is among 0-9 which correspond to the ten runs with different initial PL set. In the file, the _i_-th data point corresponds to the F1-score of the selected model (if multiple models are selected, we use their average F1-scores) when the number of queried labels is _i_.
- User interaction response time files. The naming format is "\[dataset\]\_avgTimeBetweenQueries_rand10\_\[ST-1NN seed ID\]\_.txt", for example "GunPointMaleVersusFemale_avgTimeBetweenQueries_rand10_3.txt". The single data point in this file corresponds to the average user interaction response time for the current ST-1NN seed.
