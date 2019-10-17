function genST1NNSeeds(dataset, ST1NNSeedPath, datasetPath)
% dataset: name of dataset
% ST1NNSeedPath: output path
% datasetPath: path to the dataset folder

pLabel = 1;
numLabeled = 1;

%load the dataset
data = load(fullfile(datasetPath, dataset, [dataset, '_TRAIN.tsv']));

%labels
labels = data(:, 1);
pInds = (find(labels == pLabel))';
numP = length(pInds);
numSeeds = min(numP, 10);

seeds = zeros(numSeeds, numLabeled);
for i = 1 : numSeeds
    seed = pInds(sort(random('unid', numP, 1, numLabeled)));
	while(ismember(seed, seeds, 'rows') || length(unique(seed)) < numLabeled)
		seed = pInds(sort(random('unid', numP, 1, numLabeled)));
	end
	seeds(i, :) = seed;
end

fName = fullfile(ST1NNSeedPath, ['seeds_', dataset, '_', num2str(numSeeds), '_', num2str(numLabeled) '.txt']);
fid = fopen(fName, 'w');
for i = 1 : numSeeds
    for j = 1 : numLabeled
        fprintf(fid, '%d ', seeds(i, j));
    end
    fprintf(fid, '\n');
end
fclose(fid);
