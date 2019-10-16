#include <vector>
#include "calcUtilities.h"
#pragma once

class STTreeNode
{
public:
	int ind;
	int parentInd;
	std::vector<int> childInds;

	int numTrain;
	std::vector<int> influencedIndsByRanking;
	std::vector<int> isInfluenced;

	STTreeNode(int ind, int parentInd);
	void addChild(int newChildInd);

	STTreeNode();
	STTreeNode(int ind, int parentInd, int numTrain);
	void updateIsInfluenced(int influencedInd);
	void updateIsInfluenced(std::vector<int> influencedInds);
	void setInfluencedIndsByRanking(int *rankedInds);
	void copyTo(STTreeNode *otherNode);
};

STTreeNode::STTreeNode(int ind, int parentInd) {
	this->ind = ind;
	this->parentInd = parentInd;
	this->numTrain = 0;
}

void STTreeNode::addChild(int newChildInd) {
	if (!ismember(newChildInd, this->childInds))
		this->childInds.push_back(newChildInd);
}

STTreeNode::STTreeNode() {
}

STTreeNode::STTreeNode(int ind, int parentInd, int numTrain) {
	this->ind = ind;
	this->parentInd = parentInd;
	this->numTrain = numTrain;
	isInfluenced.resize(numTrain);
	memset(&isInfluenced[0], 0, numTrain * sizeof(int));
}

void STTreeNode::updateIsInfluenced(int influencedInd) {
	isInfluenced[influencedInd] = 1;
}

void STTreeNode::updateIsInfluenced(std::vector<int> influencedInds) {
	for (int i = 0; i < influencedInds.size(); i++) {
		isInfluenced[influencedInds[i]] = 1;
	}
}

void STTreeNode::setInfluencedIndsByRanking(int* rankedInds) {
	int curInd;
	for (int i = 0; i < numTrain; i++) {
		curInd = rankedInds[i];
		if(isInfluenced[curInd])
			influencedIndsByRanking.push_back(curInd);
	}
}

void STTreeNode::copyTo(STTreeNode* otherNode) {
	otherNode->ind = ind;
	otherNode->parentInd = parentInd;
	otherNode->numTrain = numTrain;
	otherNode->childInds = childInds;
	otherNode->influencedIndsByRanking = influencedIndsByRanking;
	otherNode->isInfluenced = isInfluenced;
}