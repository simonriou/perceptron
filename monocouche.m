clear
close all
clc

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
P = length(digits); % Nb perceptrons = nb de digits

% Menu - choix entraînement ou test de la monocouche
disp("ATTENTION à bien tester les données sur le bon entraînement !");
disp('Choisissez une option :');
disp('1. Entraînement');
disp('2. Test');
disp('3. Quitter');
choice = input('Entrez votre choix (1, 2 ou 3) : ');

if choice == 1
    disp('Lancement de l''entraînement...');
    trainModel(digits, P);
elseif choice == 2
    disp('Lancement du test...');
    testModel(digits, P);
elseif choice == 3
    disp('Arrêt du script.');
    return;
else
    disp('Choix non valide. Veuillez redémarrer et entrer 1, 2 ou 3.');
end

function trainModel(digits, P)
    %% Initialisation des données
    
    rawImgs = [];
    inputLabels = [];
    
    for p = 1:P
        digit = digits(p);
        
        % Données d'entraînement
        trainData = load(sprintf("Data/DigitTest_%d.mat", digit));
        rawImgs = cat(3, rawImgs, trainData.imgs);
        inputLabels = [inputLabels ; trainData.labels];
    end
    
    inputLabels = inputLabels';
    
    N = size(rawImgs, 3); % Nb de données d'entrapinement
    ImgSize = size(rawImgs, 1) * size(rawImgs, 2); % Taille des images
    
    %% Mise en forme des données d'entrée
    shapedImgs = reshape(rawImgs, ImgSize, N); % Format vecteurs colonnes
    
    biasesX = ones(1, N); % Des 1 pour le biais
    shapedImgs = [biasesX ; shapedImgs];
    
    w = zeros(ImgSize+1, P)*0.01; % Poids initiaux
    
    %% Paramètres de la descente de gradient
    rho = 0.1;
    Nb = 500;
    grad = zeros(Nb, ImgSize+1);
    f = zeros(Nb, 1);
        
    for p = 1:P % Pour chaque perceptron de la couche
        % Mise en forme des labels
        pLabels = zeros(1, N);
    
        for i = 1:N
            if inputLabels(1, i) == digits(p)
                pLabels(1, i) = 1;
            end
        end
    
        % Initialisation de la descente
        Y = 1./(1+exp(-(shapedImgs' * w(:,p))'));
        S = (Y - pLabels).^2;
        f(1) = 1/(2*Nb) * sum(S, 2);
        grad(1,:) = (Y - pLabels).*(Y - Y.^2)*shapedImgs';
    
        % Descente de gradient
        for i = 2:Nb
            Y = 1./(1+exp(-(shapedImgs' * w(:,p))'));
            S = (Y - pLabels).^2;
            f(i) = 1/(2*Nb) * sum(S, 2); % Mise à jour du critère
    
            grad(i,:) = shapedImgs * ((Y - pLabels).*Y.*(1-Y)).'/Nb; % Mise à jour du gradient
    
            w(:,p) = w(:,p) - rho * grad(i, :)'; % Mise à jour des poids
        end
        
        save('modele_perceptron.mat', 'w');
    end
    disp("Réseau entraîné.");
end

function testModel(digits, P)
    %% Initialiation des données de test
    testImgs = [];
    testLabels = [];
    
    for p = 1:P
        digit = digits(p);
       
        % Données de test
        testData = load(sprintf("Data/DigitTest_%d.mat", digit));
        testImgs = cat(3, testImgs, testData.imgs);
        testLabels = [testLabels ; testData.labels];
    end
    testLabels = testLabels';
    NTest = size(testImgs, 3); % Nb de données de test
    ImgSize = size(testImgs, 1) * size(testImgs, 2); % Taille des images
    
    shapedTestImgs = reshape(testImgs, ImgSize, NTest);
    biasesXTest = ones(1, NTest);
    shapedTestImgs = [biasesXTest ; shapedTestImgs];

    wData = load("modele_perceptron.mat");
    w = wData.w;

    %% Calcul de la sortie de la couche
    outputTestLabels = zeros(P, NTest);
    
    for p = 1:P
        ZP = shapedTestImgs' * w(:,p);
        YP = 1./(1+exp(-ZP));
        outputTestLabels(p, :) = YP';
    end
    
    % Attribution des classes de sortie
    for i = 1:NTest % Pour chaque jeu de données
        [~, imax] = max(outputTestLabels(:,i)); % On trouve l'indice de la plus grande valeur
        outputTestLabels(:, i) = 0; % On met la colonne à 0
        outputTestLabels(imax, i) = 1; % On met la plus grande à 1
    end
    
    finalLabels = zeros(1, NTest);
    
    % On remet en forme les labels avec les digits
    for i = 1:NTest
        idx = outputTestLabels(:,i) == 1;
        finalLabels(1, i) = digits(idx);
    end
    
    %% Matrice de confusion
    confM = confusionmat(testLabels, finalLabels);
    f3 = figure('Name', 'Matrice de confusion');
    confusionchart(confM)

    disp("Test terminé.");
end
