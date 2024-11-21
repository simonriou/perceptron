clear
close all
clc

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
P = length(digits); % Nb of output neurons = nb of classes

% Menu - choose between training and testing
disp("BE CAREFUL to test the model on the right weights!");
disp('Choose an option :');
disp('1. Training');
disp('2. Test');
disp('3. Close');
choice = input('Your choice (1, 2 or 3): ');

if choice == 1
    disp('Beginning training...');
    trainModel(digits, P);
elseif choice == 2
    disp('Beginning testing...');
    testModel(digits, P);
elseif choice == 3
    disp('Killing script.');
    return;
else
    disp('Invalid choice. Please restart the program and choose between 1, 2 or 3.');
end

function trainModel(digits, P)
    %% Data loading
    
    rawImgs = [];
    inputLabels = [];
    
    for p = 1:P
        digit = digits(p);
        
        % Training Data
        trainData = load(sprintf("Data/DigitTest_%d.mat", digit));
        rawImgs = cat(3, rawImgs, trainData.imgs);
        inputLabels = [inputLabels ; trainData.labels];
    end
    
    inputLabels = inputLabels';
    
    N = size(rawImgs, 3); % Size of the training set
    ImgSize = size(rawImgs, 1) * size(rawImgs, 2); % Images size
    
    %% Resizing input images
    shapedImgs = reshape(rawImgs, ImgSize, N); % Column vectors
    
    biasesX = ones(1, N); % Biases
    shapedImgs = [biasesX ; shapedImgs];
    
    w = zeros(ImgSize+1, P)*0.01; % Initial weights
    
    %% Gradient descent parameters
    rho = 0.1;
    Nb = 500;
    grad = zeros(Nb, ImgSize+1);
    f = zeros(Nb, 1);
        
    for p = 1:P % For each perceptron
        % Label formatting
        pLabels = zeros(1, N);
    
        for i = 1:N
            if inputLabels(1, i) == digits(p)
                pLabels(1, i) = 1;
            end
        end
    
        % Descent initialization
        Y = 1./(1+exp(-(shapedImgs' * w(:,p))'));
        S = (Y - pLabels).^2;
        f(1) = 1/(2*Nb) * sum(S, 2);
        grad(1,:) = (Y - pLabels).*(Y - Y.^2)*shapedImgs';
    
        % Gradient descent
        for i = 2:Nb
            Y = 1./(1+exp(-(shapedImgs' * w(:,p))'));
            S = (Y - pLabels).^2;
            f(i) = 1/(2*Nb) * sum(S, 2); % Loss update
    
            grad(i,:) = shapedImgs * ((Y - pLabels).*Y.*(1-Y)).'/Nb; % Gradient update
    
            w(:,p) = w(:,p) - rho * grad(i, :)'; % Weight update
        end
        
        save('modele_perceptron.mat', 'w');
    end
    disp("Model successfully trained.");
end

function testModel(digits, P)
    %% Test data loading
    testImgs = [];
    testLabels = [];
    
    for p = 1:P
        digit = digits(p);
       
        % Test set
        testData = load(sprintf("Data/DigitTest_%d.mat", digit));
        testImgs = cat(3, testImgs, testData.imgs);
        testLabels = [testLabels ; testData.labels];
    end
    testLabels = testLabels';
    NTest = size(testImgs, 3); % Test data size
    ImgSize = size(testImgs, 1) * size(testImgs, 2); % Images size
    
    shapedTestImgs = reshape(testImgs, ImgSize, NTest);
    biasesXTest = ones(1, NTest);
    shapedTestImgs = [biasesXTest ; shapedTestImgs];

    wData = load("modele_perceptron.mat");
    w = wData.w;

    %% Ouput computation
    outputTestLabels = zeros(P, NTest);
    
    for p = 1:P
        ZP = shapedTestImgs' * w(:,p);
        YP = 1./(1+exp(-ZP));
        outputTestLabels(p, :) = YP';
    end
    
    % Class labelization
    for i = 1:NTest % For each data set
        [~, imax] = max(outputTestLabels(:,i)); % Get the indice of the greatest
        outputTestLabels(:, i) = 0; % Set the column to 0
        outputTestLabels(imax, i) = 1; % And set the greatest to 1
    end
    
    finalLabels = zeros(1, NTest);
    
    % Label formatting
    for i = 1:NTest
        idx = outputTestLabels(:,i) == 1;
        finalLabels(1, i) = digits(idx);
    end
    
    %% Confusion matrix
    confM = confusionmat(testLabels, finalLabels);
    f3 = figure('Name', 'Confusion matrix');
    confusionchart(confM)

    disp("Test done.");
end
