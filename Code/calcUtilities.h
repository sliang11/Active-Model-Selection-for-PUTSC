#pragma once
#include <float.h>
#include <math.h>
#include <vector>
#include <algorithm>

//ref: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
bool doubleIsEqual(double x, double y,
	double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON)
{
	//Comparison using an absolute value
	double diff = fabs(x - y);
	if (diff <= maxDiff)
		return true;

	//Comparison using a relative value
	x = fabs(x);
	y = fabs(y);
	double largest = (y > x) ? y : x;

	if (diff <= largest * maxRelDiff)
		return true;
	return false;
}

bool doubleIsZero(double x, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	return doubleIsEqual(x, 0.0, maxDiff, maxRelDiff);
}

//x <= y
bool doubleLeq(double x, double y, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	return (doubleIsEqual(x, y, maxDiff, maxRelDiff)) || (x < y);

}

//x >= y
bool doubleGeq(double x, double y, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	return (doubleIsEqual(x, y, maxDiff, maxRelDiff)) || (x > y);
}

int sign(int x) {
	int ret;
	if (x > 0)
		ret = 1;
	else if (x == 0)
		ret = 0;
	else
		ret = -1;
	return ret;
}

int sign(double x, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	if (doubleIsZero(x, maxDiff, maxRelDiff))
		return 0;
	if (x > 0)
		return 1;
	if (x < 0)
		return -1;
}

template <class T>
T min(T x, T y){
	if (x < y)
		return x;
	return y;
}

template <class T>
T max(T x, T y){
	if (x > y)
		return x;
	return y;
}

template <class T>
T min(T* x, int numElm, int stride = 1){
	T val, ret = x[0];
	for (int i = 1; i < numElm; i++){
		val = x[i * stride];
		if (val < ret)
			ret = val;
	}
	return ret;
}

template <class T>
T max(T* x, int numElm, int stride = 1){
	T val, ret = x[0];
	for (int i = 1; i < numElm; i++){
		val = x[i * stride];
		if (val > ret)
			ret = val;
	}
	return ret;
}

template <class T>
void min(T &minVal, int &minIdx, T* x, int numElm, int stride = 1) {
	T val;
	minVal = x[0];
	minIdx = 0;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];
		if (val < minVal) {
			minVal = val;
			minIdx = i;
		}
	}
}

template <class T>
void max(T &maxVal, int &maxIdx, T* x, int numElm, int stride = 1) {
	T val;
	maxVal = x[0];
	maxIdx = 0;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];
		if (val > maxVal) {
			maxVal = val;
			maxIdx = i;
		}
	}
}

void minWithTies(int &minVal, int *minIdxes, int &numMin, int* x, int numElm, int stride = 1) {
	int val;
	minVal = x[0];
	minIdxes[0] = 0;
	numMin = 1;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];
		if (val < minVal) {
			minVal = val;
			minIdxes[0] = i;
			numMin = 1;
		}
		else if (val == minVal) {
			minIdxes[numMin] = i;
			numMin++;
		}
	}
}

void minWithTies(double& minVal, int* minIdxes, int& numMin, double* x, int numElm, 
	int stride = 1, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	double val;
	minVal = x[0];
	minIdxes[0] = 0;
	numMin = 1;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];

		if (doubleIsEqual(val, minVal, maxDiff, maxRelDiff)) {
			minIdxes[numMin] = i;
			numMin++;
		}
		else if (val < minVal) {
			minVal = val;
			minIdxes[0] = i;
			numMin = 1;
		}
	}
}

void maxWithTies(int &maxVal, int *maxIdxes, int &numMax, int* x, int numElm, int stride = 1) {
	int val;
	maxVal = x[0];
	maxIdxes[0] = 0;
	numMax = 1;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];
		if (val > maxVal) {
			maxVal = val;
			maxIdxes[0] = i;
			numMax = 1;
		}
		else if (val == maxVal) {
			maxIdxes[numMax] = i;
			numMax++;
		}
	}
}

void maxWithTies(double& maxVal, int* maxIdxes, int& numMax, double* x, int numElm, 
	int stride = 1, double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON) {
	double val;
	maxVal = x[0];
	maxIdxes[0] = 0;
	numMax = 1;
	for (int i = 1; i < numElm; i++) {
		val = x[i * stride];

		if (doubleIsEqual(val, maxVal, maxDiff, maxRelDiff)) {
			maxIdxes[numMax] = i;
			numMax++;
		}
		else if (val > maxVal) {
			maxVal = val;
			maxIdxes[0] = i;
			numMax = 1;
		}
	}
}

template <class T>
T sum(T *x, int numElm, int stride = 1){
	T total = 0;
	for (int i = 0; i < numElm; i++) {
		total += x[i * stride];
	}
	return total;
}

template <class T>
T sum2(T *x, int numElm, int stride = 1){
	T val, total2 = 0;
	for (int i = 0; i < numElm; i++){
		val = x[i * stride];
		total2 += val * val;
	}
	return total2;
}

template <class T>
T mean(T* x, int numElm, int stride = 1){
	return sum(x, numElm, stride) / numElm;
}

double var(double* x, int numElm, int stride = 1, 
	double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON){	
	double total2 = sum2(x, numElm, stride);
	double total = sum(x, numElm, stride);
	double varVal = total2 / numElm - (total * total) / (numElm * numElm);	
	if (doubleIsZero(varVal, maxDiff, maxRelDiff) || varVal < 0)	
		varVal = 0;
	return varVal;
}

double stdv(double* x, int numElm, int stride = 1, 
	double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON){
	return sqrt(var(x, numElm, stride, maxDiff, maxRelDiff));
}

void zscore(double *zx, double *x, int numElm, int zstride = 1, int stride = 1,
	double maxDiff = DBL_EPSILON, double maxRelDiff = DBL_EPSILON){
	double avg = mean(x, numElm, stride);
	double stdev = stdv(x, numElm, stride, maxDiff, maxRelDiff);
	if (doubleIsZero(stdev, maxDiff, maxRelDiff))
		stdev = 1;
	for (int i = 0; i < numElm; i++)
		zx[i * zstride] = (x[i * stride] - avg) / stdev;
}

bool ismember(int x, std::vector<int> v) {
	return std::find(v.begin(), v.end(), x) != v.end();
}

bool isequal(int* x, int* y, int numElm) {
	for (int i = 0; i < numElm; i++) {
		if(x[i] != y[i])
			return false;
	}
	return true;
}