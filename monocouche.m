clear
close all
clc

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
P = length(digits); % Number of perceptrons = number of digits

% Menu - choose training or testing for the single layer network
disp("WARNING: Make sure to test the data on the correct training!");
disp('Choose an option:');
disp('1. Training');
disp('2. Testing');
disp('3. Quit');
choice = input('Enter your choice (1, 2 or 3): ');

if choice == 1
    disp('Starting training...');
    trainModel(digits, P);
elseif choice == 2
    disp('Starting testing...');
    testModel(digits, P);
elseif choice == 3
    disp('Exiting script.');
    return;
else
    disp('Invalid choice. Please restart and enter 1, 2, or 3.');
end

function trainModel(digits, P)
    %% Data Initialization
    
    rawImgs = [];
    inputLabels = [];
    
    for p = 1:P
        digit = digits(p);
        
        % Training data
        trainData = load(sprintf("Data/DigitTest_%d.mat", digit));
        rawImgs = cat(3, rawImgs, trainData.imgs);
        inputLabels = [inputLabels ; trainData.labels];
    end
    
    inputLabels = inputLabels';
    
    N = size(rawImgs, 3); % Number of training data
    ImgSize = size(rawImgs, 1) * size(rawImgs, 2); % Image size
    
    %% Reshape input data
    shapedImgs = reshape(rawImgs, ImgSize, N); % Column vector format
    
    biasesX = ones(1, N); % Ones for the bias
    shapedImgs = [biasesX ; shapedImgs];
    
    w = zeros(ImgSize+1, P)*0.01; % Initial weights
    
    %% Gradient descent parameters
    rho = 0.1;
    Nb = 500;
    grad = zeros(Nb, ImgSize+1);
    f = zeros(Nb, 1);
        
    for p = 1:P % For each perceptron in the layer
        % Format labels
        pLabels = zeros(1, N);
    
        for i = 1:N
            if inputLabels(1, i) == digits(p)
                pLabels(1, i) = 1;
            end
        end
    
        % Gradient descent initialization
        Y = 1./(1+exp(-(shapedImgs' * w(:,p))'));
        S = (Y - pLabels).^2;
        f(1) = 1/(2*Nb) * sum(S, 2);
        grad(1,:) = (Y - pLabels).*(Y - Y.^2)*shapedImgs';
    
        % Gradient​⬤
