clear
close all

digits = [1, 8];
P = length(digits);
Nb = 100; % Iteration n

%% Initialisation
rawImgs = [];
inputLabels = [];
testImgs = [];
testLabels = [];
    
for p = 1:P
    digitP = digits(p);
    
    % Training data
    trainData = load(sprintf("Data/DigitTest_%d.mat", digitP));
    rawImgs = cat(3, rawImgs, trainData.imgs);
    inputLabels = [inputLabels ; trainData.labels];

    % Test data
    testData = load(sprintf("Data/DigitTest_%d.mat", digitP));
    testImgs = cat(3, testImgs, testData.imgs);
    testLabels = [testLabels ; testData.labels];
end

inputLabels = inputLabels';
testLabels = testLabels';

N = size(rawImgs, 3); % Nb of training data
NTest = size(testImgs, 3); % Nb of test data
ImgSize = size(rawImgs, 1) * size(rawImgs, 2); % Image size

% Labels formating
labels = zeros(1, N);
for i = 1:N
    if inputLabels(1, i) == digits(1)
        labels(i) = 1;
    end
end

labels_test = zeros(1, NTest);
for i = 1:NTest
    if testLabels(1, i) == digits(1)
        labels_test(i) = 1;
    end
end

%% Mise en forme des données d'entrée
shapedImgs = reshape(rawImgs, ImgSize, N); % Column vectors formating
shapedTestImgs = reshape(testImgs, ImgSize, NTest);

biasesX = ones(1, N); % Biases
biasesXTest = ones(1, NTest);
shapedImgs = [biasesX ; shapedImgs];
shapedTestImgs = [biasesXTest ; shapedTestImgs];

w1 = zeros(ImgSize+1, 1); % Initial weights

%% Paramètres de la descente de gradient

rho = 0.1;
Nb_plot = 1:Nb;

grad = zeros(Nb, ImgSize+1);
grad_plot = zeros(1, Nb);
f = zeros(Nb, 1);

Y = 1./(1+exp(-(shapedImgs' * w1)'));
S = (Y - labels).^2;
f(1) = 1/(2*Nb) * sum(S, 2);
grad(1,:) = (Y - labels).*(Y - Y.^2)*shapedImgs';

%% Itérations de la descente de gradient
for i = 2:Nb
    Y = 1./(1+exp(-(shapedImgs' * w1)'));
    S = (Y - labels).^2;
    f(i) = 1/(2*Nb) * sum(S, 2);

    grad(i,:) = shapedImgs * ((Y - labels).*Y.*(1-Y)).'/Nb; % Updating gradient
    grad_plot(1, i) = sqrt(grad(i, 1)^2 + grad(i, 2)^2 + grad(i, 3)^2); % kth iteration norm
    
    w1 = w1 - rho * grad(i, :)'; % Updating weights
end

%% Vérification
classesTrained = zeros(2, N); % Class-comparaison vector
classesTrained(2,:) = labels;
for i = 1:N
    if Y(1, i) >= 0.5
        classesTrained(1, i) = 1;
    else
        classesTrained(1, i) = 0;
    end
end

%% Visualisation
f1=figure('Name', "Log. of the criteria for iteration number.");
plot(Nb_plot, log(f));
hold on
xlabel('Itérations')
ylabel("Critère")
title("Log. of the criteria for iteration number.")

f2=figure('Name', "Log. of the gradient norm for iteration number.");
plot(Nb_plot, log(grad_plot), "r");
hold on
xlabel('Itérations')
ylabel("Norme du gradient")
title("Log. of the gradient norm for iteration number.")

% Données de test
ZTest = shapedTestImgs' * w1;
YTest = 1./(1+exp(-ZTest));
classesTest = zeros(2, NTest);
classesTest(2,:) = labels_test;
for i = 1:NTest
    if YTest(i, 1) >= 0.5
        classesTest(1, i) = 1;
    else
        classesTest(1, i) = 0;
    end
end

%% Taux d'erreur
NbErrTrain = length(find(classesTrained(1,:) - classesTrained(2,:)));
tauxTrain = 100 * NbErrTrain / N;

NbErrTest = length(find(classesTest(1,:) - classesTest(2,:)));
tauxTest = 100 * NbErrTest / NTest;

fprintf("Err. %% during the test : %f%%\n", tauxTest)

%% Matrice de confusion
confM = confusionmat(classesTest(2,:), classesTest(1, :));
f3 = figure('Name', 'Confusion matrix');
confusionchart(confM)
