
%% House Keeping

clc; clear all; close all;

%% Loading Data

load('Biobot_Training.mat');

%% Initializing

feat = F_train.f; time = F_train.t; gt = gtLabel_train;

%% Handling NaN Values

[m,n] = size(feat);

for i = 1:m
    if sum(isnan(feat(i,:))) ~= 0
        for j=1:n
            if (isnan(feat(i,j)) == 1)
                feat(i,j) = feat(i-1,j);
            end
        end
    end    
end

%% Setting the data

data = [ time feat gt ]; val = 1:m;

%% 5 fold cross validation for Fine KNN

rng default;
ind = crossvalind('Kfold', val, 5 );

a1 = []; a2 = []; a3 = []; a4 = []; a5 = [];

for i=1:m
    if ind(i) == 1
        a1 = [ a1 i ];
    end
    if ind(i) == 2
        a2 = [ a2 i ];
    end
    if ind(i) == 3
        a3 = [ a3 i ];
    end
    if ind(i) == 4
        a4 = [ a4 i ];
    end
    if ind(i) == 5
        a5 = [ a5 i ];
    end
end

b1 = trainKNN( data, [ a2 a3 a4 a5 ], a1 );
b2 = trainKNN( data, [ a1 a3 a4 a5 ], a2 );
b3 = trainKNN( data, [ a1 a2 a4 a5 ], a3 );
b4 = trainKNN( data, [ a1 a2 a3 a5 ], a4 );
b5 = trainKNN( data, [ a1 a2 a3 a4 ], a5 );

%% Compiling the KNN op

op = zeros(m,1); op(a1) = b1;
op(a2) = b2; op(a3) = b3;
op(a4) = b4; op(a5) = b5;

%% F1 for Fine KNN

c = confusionmat(gt,op);
tp = zeros(4,1); fp = zeros(4,1); fn = zeros(4,1);
for i=1:4
    for j = 1:4
        if i==j
            tp(i) = c(i,j);
        else
            fp(j) = fp(j) + c(i,j);
            fn(i) = fn(i) + c(i,j);
        end    
    end
end

p = sum(tp./(tp+fp));
r = sum(tp./(tp+fn));

F1 = p*r/(2*(p+r));

disp('Confusion Matrix for Fine KNN');
disp(c);
disp('F1 Score for Fine KNN');
disp(F1);


%% Setting i/p for HMM

N = round(.75*m); % 75% data for training, 25% for testing

op = op + 1; gt = gt + 1; % HMM accepts only classes from 1
train = op(1:N); test = op(N+1:end); % setting training and testing data

%% Running HMM

[TRANS,EMIS] = hmmestimate(op,gt);
[ESTTR,ESTEMIT] = hmmtrain(train',TRANS,EMIS);
PSTATES = hmmdecode(test',ESTTR,ESTEMIT);

%% Compiling results of HMM

res = [];
for i=1:size(PSTATES,2)
    [~,in] = max(PSTATES(:,i));
    res = [ res in ];
end

%% F1 for HMM

c = confusionmat(gt(N+1:end),res);
tp = zeros(4,1); fp = zeros(4,1); fn = zeros(4,1);
for i=1:4
    for j = 1:4
        if i==j
            tp(i) = c(i,j);
        else
            fp(j) = fp(j) + c(i,j);
            fn(i) = fn(i) + c(i,j);
        end    
    end
end

p = sum(tp./(tp+fp));
r = sum(tp./(tp+fn));

F1 = p*r/(2*(p+r));

disp('Confusion Matrix for HMM');
disp(c);
disp('F1 Score for HMM');
disp(F1);