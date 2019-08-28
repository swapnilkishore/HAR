function result = msvm(dataTraining,train_target,dataTesting) 
% This is for training of SVM
options = optimset('fmincon');
options.MaxFunEvals = 30000;
options.MaxIter=1000000000000000;
u=unique(train_target);
numClasses=length(u);
%result = zeros(length(dataTesting(:,1)),100);
k=1;
%build models
while (k<=numClasses)
    %Vectorized statement that binarizes Group
    %where 1 is the current class and 0 is all other classes
    G1vAll=(train_target==u(k));
    models(k) = svmtrain(dataTraining,G1vAll,'Kernel_Function','rbf','BoxConstraint', 1,'quadprog_opts' ,options);
    k=k+1;
    display(k);
end
% This is for Testing purpose
numClasses=6;
result=zeros(length(dataTesting(:,1)),1);
k=1;
for j=1:size(dataTesting,1)
    for k=1:numClasses
    if(svmclassify(models(k),dataTesting(j,:))) 
        result(j)=k;
        break;
   % else 
    %    result1(j,k)=0;
        %break;
        end
    end
   end