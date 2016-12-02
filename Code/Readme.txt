ECE 792 - Applications of Graphs and Graphical Modelling

Project 2.2 : Biobot Motion Classification using HMM

Team : Anamika Thakur, Jordan Campbell and Shankara Narayanan Sethuraman

This folder contains a runClassifier.m file and two .mat file containing the trained Fine KNN CLassifier and the Trained HMM.

The classifiers has been trained with 43 features : 1st feature F_train.t followed by F_train.f

The runClassifier function take input FeatureList which is a NxD matrix where N is the number of samples used for testing and D is the number of features per sample.

The function will be able to test only if D = 43 so kindly provide the input in the same format.