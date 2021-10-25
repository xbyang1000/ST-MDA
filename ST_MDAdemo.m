function ST_MDAdemo()
%demo for ST-MDA
filename = 'Video2Nofire4.jpg'; % non-fire target scene
Im = imread(filename);
[row,col,~] = size(Im);
[label,num] = superpixels(Im,floor(row*col/100));% superpixel samples

[Sample,~] = PixelSample(Im,label); % get samples
Label = label(:);
% show superpixel segmentation
% figure;hold on;  
% BW = boundarymask(label);
% imshow(imoverlay(Im,BW,'cyan'));

SuperSample = zeros(num,3);
for i=1:num
     SuperSample(i,:) = mean(Sample(Label==i,:));
end

X = SuperSample;
D = pdist(X);
D = squareform(D);%distance matrix
W = zeros(size(D));
for i = 1:size(D,1), D(i,i)=inf; end
k_num = 3;
%% compute similarity matrix 
for i = 1:size(X,1)
    [value,posi] = sort(D(:,i));
    KNN = posi(1:k_num,1);
    for j=1:k_num
        W(i,KNN(j)) = X(i,:)*X(KNN(j),:)'/( norm(X(i,:)) * norm(X(KNN(j),:)) );        
    end
end
W = (W+W')/2; % avoid sigularity for symmetric matrix
D = diag(sum(W)); % redefine D
D = inv(D);
D = sqrt(D);
L =  D*W*D;

load Class12Image;% load source classifiers, saved as "Class12Image.mat"

FixSampleNum = 2000;
%% compute P for scene matching, named gamma
s = size(L2SVM_DeciPlane,2);
X = SuperSample;
clear SuperSample;
ClassNum = size(X,1);
F = zeros(ClassNum,s);
for i=1:s
   F(:,i) = X*L2SVM_DeciPlane(1:3,i)+repmat(L2SVM_DeciPlane(4,i),ClassNum,1);
end
AMat = F'*L*F;
AMat = (AMat+AMat')/2;

%% Solve P for scene matching
options = optimoptions(@quadprog,'Algorithm','interior-point-convex','display','none');
f = zeros(s,1); Aeq = ones(1,s);beq = 1;
lb = zeros(s,1);ub = ones(s,1)*inf;
[Gamma,fval,exitflag] = quadprog(AMat,f,[],[],Aeq,beq,lb,ub,[],options);
% 
Ind = randperm(size(X,1))';
X2 = X(Ind(1:FixSampleNum/2,:),:); % avoid unbalance training
epsilon = 1e-06;
ind = find(Gamma>epsilon);
SelectClf = ind';
PerSceneNum = []; X1 = []; PerSceneNum = [];Thres = [];GammThre = [];
for i = 1:length(SelectClf)
    num = SelectClf(i);
    FData = GetFireData(num); 
    % drawn positive sample by matching coefficency
    DrawNum = floor(Gamma(num)*FixSampleNum/2);
    Ind = randperm(size(FData,1)); Ind = Ind';
    if (size(FData,1)<DrawNum)
        TmpSample = FData;
        PerSceneNum = [PerSceneNum size(FData,1)];
        X1 = [X1; FData];
    else
        PerSceneNum = [PerSceneNum DrawNum];
        TmpSample   = FData(1:DrawNum,:);
        X1 = [X1; FData(1:DrawNum,:)];        
    end
    Thres = [Thres; Gamma(num)*TmpSample*L2SVM_DeciPlane(1:3,num)+repmat(L2SVM_DeciPlane(4,num),size(TmpSample,1),1)];
    GammThre = [GammThre; Gamma(num)*ones(PerSceneNum(i),1)];
end
% 
F = zeros(size(X1,1),length(SelectClf));
for i = 1:length(SelectClf)
    F(:,i) = X1*L2SVM_DeciPlane(1:3,SelectClf(i))+repmat(L2SVM_DeciPlane(4,SelectClf(i)),size(X1,1),1);
end
% 
 PosiThre = GammThre + F*Gamma(SelectClf',1) - Thres; % Get \rho
 TransferClassifier = TrainingClf(X1,X2,PosiThre); % training ST-MDA

TestIm = imread('Zenmuse2_2.jpg'); %load test fire images T2
TestResult = PredictClf(TransferClassifier,TestIm);
figure(1); hold on;
imshow(TestIm); % show T2
figure(2);hold on;
imshow(TestResult.Im); % show  detected fire result
end

function TestResult = PredictClf(TransferClassifier,Im)
  [row,col,~] = size(Im);
  [TestSample,~] = PixelSample(Im,zeros(10,1));
  w = TransferClassifier.w;
  b = TransferClassifier.b;
  predict = TestSample*w + repmat(b,row*col,1);
  Ind = find(predict>0);
  ReIm = zeros(row*col,3);
  ReIm(Ind,:) = TestSample(Ind,:);
  R = ReIm(:,1);  G = ReIm(:,2);  B = ReIm(:,3);
  R = reshape(R,[row col]);
  G = reshape(G,[row col]);
  B = reshape(B,[row col]);
  TestResult.Im = uint8(cat(3,R,G,B));
end
function TransClassifier = TrainingClf(X1,X2,PosiThre)
    n1 = size(X1,1); n2 = size(X2,1);
    n  = n1+n2;     d  = size(X1,2);
    Y1 = ones(n1,1); Y2 = -ones(n2,1);
    Y  = [Y1;Y2];
    TuningX = [X1;X2];
    TuningY = [ones(n1,1);-ones(n2,1)];
    
    Correct = 0; C0 = 0; 
    L2SVM = [];
    options = optimoptions(@quadprog,'Algorithm','interior-point-convex','Display', 'none');
    H = zeros(d+1+n);  
    for i=1:d
        H(i,i) = 1;
    end
    A = [-X1 -ones(n1,1) -eye(n1) zeros(n1,n2)];
    A = [A; X2 ones(n2,1) zeros(n2,n1) -eye(n2)];
    bb = [PosiThre;-ones(n2,1)];%%%
    lb = [-ones(d+1,1)*inf; zeros(n,1)];
    ub = ones(d+1+n,1)*inf;
    %D = diag(Y); 
    for i=-4:4
        C = 10^i;C1 = C*10;
        f = [zeros(d+1,1);C1*ones(n1,1);C*ones(n2,1)];
        %A = [-D*X -Y -eye(n)];
        [solution,fval,exitflag] = quadprog(H,f,A,bb,[],[],lb,ub,[],options);
        w = solution(1:d,1);
        b = solution(d+1,1);
        predict = TuningX*w + repmat(b,size(TuningX,1),1);
        predict = sign(predict);
        corr = sum(predict==TuningY)/size(TuningX,1);
        if (corr>= Correct)
              Correct = corr;  C0 = C;
        end 
    end
    
    C = C0;C1 = C*10;
   f = [zeros(d+1,1);C1*ones(n1,1);C*ones(n2,1)];
 tic;
 [solution,fval,exitflag] = quadprog(H,f,A,bb,[],[],lb,ub,[],options);
 L2SVM.Trtime = toc;
 w = solution(1:d,1);
 b = solution(d+1,1);
 predict1 = X1*w + repmat(b,size(X1,1),1);  predict1 = sign(predict1);
 predict2 = X2*w + repmat(b,size(X2,1),1);  predict2 = sign(predict2);

L2SVM.C = C;
L2SVM.TrTP = sum(predict1==Y1)/n1;
L2SVM.TrFP = sum(predict2~=Y2)/n2;
L2SVM.w = w;
L2SVM.b = b;
TransClassifier = L2SVM;
end
function FireData = GetFireData(num)
     filename = strcat('fire',num2str(num),'.jpg');
     Im = imread(filename);    
     MatFile = strcat(num2str(num),'.mat');
     load(MatFile);
     Y = double(Label);
    [X,~] = PixelSample(Im,Label);
    X = X(Label,:);
    FireData = X;
end
function [X,Y] = PixelSample(Im,label)
Y = label;
R = Im(:,:,1);  G =  Im(:,:,2);   B = Im(:,:,3);
R = R(:);  G = G(:);   B = B(:);
X = double([R G B]);
end