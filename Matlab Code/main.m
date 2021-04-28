
[~, ~, Finadata] = xlsread('microsoft_historical_finance_data_train.csv');
Finadata(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),Finadata)) = {''};


count=1;

for i= 2:250
%disp(i);
Dates{i,1}=Finadata{i,1};
PRICE_OPEN(i,1)=Finadata{i,2};
PRICE_HIGH(i,1)=Finadata{i,3};
PRICE_LOW(i,1)=Finadata{i,4};
PRICE_CLOSE(i,1)=Finadata{i+1,5};
end

data=double([PRICE_OPEN PRICE_LOW PRICE_HIGH PRICE_CLOSE])/100000;
Inputs = data(30:end,:);
Targets =(double(PRICE_OPEN(14:end)))/100000;
nData = size(Inputs,1);


PERM = randperm(nData);
pTrain=0.85;
nTrainData=round(pTrain*nData);
TrainInd=PERM(1:nTrainData);
TrainInputs=Inputs(TrainInd,:);
TrainTargets=Targets(TrainInd,:);

pTest=1-pTrain;
nTestData=nData-nTrainData;
TestInd=PERM(nTrainData+1:end);
TestInputs=Inputs(TestInd,:);
TestTargets=Targets(TestInd,:);


nCluster=5;       
Exponent=2;        
MaxIt=200;
MinImprovment=1e-7;
DisplayInfo=1;

FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];

fis=genfis3(TrainInputs,TrainTargets,'sugeno',nCluster,FCMOptions);


MaxEpoch=200;               
ErrorGoal=0;            
InitialStepSize=0.01;       
StepSizeDecreaseRate=0.9;   
StepSizeIncreaseRate=1.1;    
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=0;
% 0: Backpropagation
% 1: Hybrid
            
fis=anfis([TrainInputs TrainTargets],fis,TrainOptions,DisplayOptions,[],OptimizationMethod);

%% Apply ANFIS to Data
Outputs=evalfis(Inputs,fis);
TrainOutputs=Outputs(TrainInd,:);
TestOutputs=Outputs(TestInd,:);

%% Error Calculation
TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors.^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors.^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

%% Plot Results

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

figure;
PlotResults(TestTargets,TestOutputs,'Test Data');

figure;
PlotResults(Targets,Outputs,'All Data');

