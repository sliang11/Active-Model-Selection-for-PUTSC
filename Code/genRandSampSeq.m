function genRandSampSeq(dataset, datasetPath, ST1NNSeedPath, modelSelectSeedPath)
% dataset: name of dataset
% datasetPath: path to the dataset folder
% ST1NNSeedPath: path to the file consisting of initial PL example indices for the multiple runs of ST-1NN
% modelSelectSeedPath: output path

numRandRuns = 10;
initNumPLabeled = 1;
pLabel = 1;

data = load(fullfile(datasetPath, dataset, [dataset, '_TRAIN.tsv']));
trainLabels = data(:, 1);
numTrain = length(trainLabels);
numP = length(find(trainLabels == pLabel));

numSeeds = min(numP, 10);
seeds = load(fullfile(ST1NNSeedPath, ['seeds_', dataset, '_', num2str(numSeeds), '_1.txt']));

numU = numTrain - initNumPLabeled;
for j = 1 : numSeeds
    seed = seeds(j);
    unlabeledInds = setdiff(1 : numTrain, seed);
    
    randSeqs = zeros(numRandRuns, numU);
    for k = 1 : numRandRuns
        seq = unlabeledInds(randperm(numU));
        while ismember(seq, randSeqs, 'rows')
            seq = unlabeledInds(randperm(numU));
        end
        randSeqs(k, :) = seq;
    end
    
    fName = fullfile(modelSelectSeedPath, [dataset, '_randSampSequences_rand10_single_', ...
        num2str(numRandRuns), '_', num2str(j - 1), '.txt']);
    fid = fopen(fName, 'w');
    for k = 1 : numRandRuns
        seq = randSeqs(k, :);
        for t = 1 : numU
            fprintf(fid, '%d ', seq(t));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
