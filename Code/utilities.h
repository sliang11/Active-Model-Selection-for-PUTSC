#pragma once

#include <iostream>
#include <fstream>
#include <string.h>

#define BUF_SIZE 200000

template <class T>
void importTimeSeries(T *tss, int *labels, std::string datasetPath, std::string datasetName,
	std::string pfx, int numTs, int tsLen){	

	std::string fName = datasetPath + "/" + datasetName + "/" + datasetName + "_" + pfx + ".tsv";

	std::ifstream fin;
	fin.open(fName.c_str());
	if (!fin)
		exit(1);

	char buf[BUF_SIZE];
	char *temp;
	for (int i = 0; i < numTs; i++){
		fin.getline(buf, BUF_SIZE, '\n');
		temp = strtok(buf, " ,\r\n\t");
		labels[i] = (int)(atof(temp));
		for (int j = 0; j < tsLen; j++){
			temp = strtok(NULL, " ,\r\n\t");
			tss[i * tsLen + j] = atof(temp);
		}
	}
	fin.close();
}

void relabel(int *labels, int numTs, int pLabel){
	for (int i = 0; i < numTs; i++){
		if (labels[i] == pLabel)
			labels[i] = 1;
		else
			labels[i] = 0;
	}
}

template <class T>
void importMatrix(T *output, std::string fName, int numRow, int numCol, bool isInteger){
	std::ifstream fin;
	fin.open(fName.c_str());
	if (!fin)
		exit(1);

	char buf[BUF_SIZE];
	char *temp;
	for (int i = 0; i < numRow; i++){
		fin.getline(buf, BUF_SIZE, '\n');
		temp = strtok(buf, " ,\r\n\t");

		if (isInteger)
			output[i * numCol] = atoll(temp);
		else
			output[i * numCol] = atof(temp);
		for (int j = 1; j < numCol; j++){
			temp = strtok(NULL, " ,\r\n\t");

			if (isInteger)
				output[i * numCol + j] = atoll(temp);
			else {
				output[i * numCol + j] = atof(temp);
			}
		}
	}
	fin.close();
}

template<class T>
void exportMatrix(T* data, std::string fName, int numRow, int numCol, int precision = 0) {
	std::ofstream fout;
	if (precision != 0) {
		fout.precision(precision);
	}

	fout.open(fName.c_str());
	for (int i = 0; i < numRow; i++) {
		for (int j = 0; j < numCol; j++) {
			fout << data[i * numCol + j] << " ";
		}
		fout << std::endl;
	}
	fout.close();
}

template<class T>
T* safeMalloc(int numElm) {
	T *pt = (T *)malloc(numElm * sizeof(T));
	if (pt == NULL) {
		std::cout << "Malloc failed!" << std::endl;
		exit(1);
	}
	return pt;
}

template<class T>
void printMatrix(T* data, int numRow, int numCol, int precision = 0) {
	if (precision != 0) {
		std::cout.precision(precision);
	}

	for (int i = 0; i < numRow; i++) {
		for (int j = 0; j < numCol; j++) {
			std::cout << data[i * numCol + j] << " ";
		}
		std::cout << std::endl;
	}
}