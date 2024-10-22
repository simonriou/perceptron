clear
close all

% Data Initialization
dataTrain = load('DataSimulation/DataTrain_2Classes_Perceptron_2.mat');
x = dataTrain.data;
c = dataTrain.c;

N = size(x, 2); % Number of data sets

dataTest = load('DataSimulation/DataTest_2Classes_Perceptron_2.mat');
xT = dataTest.dataTest;
cT = dataTest.cTest;

% Create the input vector
xz = ones(1, N);
X = [xz ; x];
XT = [xz ; xT];

% Initial weight vector
w1 = randn(3, 1) * 0.01;

% Gradient descent parameters
rho = 0.01;
Nb = 1000;
Nb_plot = 1:Nb;

grad = zeros(Nb, 3);
grad_plot = zeros(1, Nb);
f = zeros(Nb, 1);

% Gradient descent iterations
for i = 1:Nb
    Y = 1./(1+exp(-(X' * w1)'));
    S = (Y - c).^2;
    f(i) = 1/(2*500) * sum(S, 2); % Update the cost function

    grad(i,:) = (Y - c).*(Y - Y.^2)*X'; % Update the gradient
    grad_plot(1, i) = sqrt(grad(i, 1)^2 + grad(i, 2)^2 + grad(i, 3)^2); % Norm at iteration k
    w1 = w1 - rho * grad(i, :)'; % Update the weights
end

% Verification
finalClasses = zeros(2, N); % Class comparison vector
finalClasses(2,:) = c;
for i = 1:N
    if Y(1, i) >= 0.5
        finalClasses(1, i) = 1;
    else
        finalClasses(1, i) = 0;
    end
end

f1=figure('Name', "Log. Cost function vs number of iterations");
plot(Nb_plot, log(f));
hold on
xlabel('Iterations')
ylabel("Cost function")
title("Log. Cost function vs number of iterations")

f2=figure('Name', "Log. Gradient norm vs number of iterations");
plot(Nb_plot, log(grad_plot), "r");
hold on
xlabel('Iterations')
ylabel("Gradient norm")
title("Log. Gradient norm vs number of iterations")

f3=figure('Name', "Initial & post-training classes");
f3.Position = [100 100 1800 900];
hold on
xlim([-12 12])
ylim([-12 12])
title("Initial & post-training classes")

plot(X(2, (finalClasses(1, :) == 0)), X(3, (finalClasses(1, :) == 0)), "ob");
plot(X(2, (finalClasses(1, :) == 1)), X(3, finalClasses(1, :) == 1), "or");

plot(X(2, (finalClasses(2, :) == 0)), X(3, (finalClasses(2, :) == 0)), "xb");
plot(X(2, (finalClasses(2, :) == 1)), X(3, finalClasses(2, :) == 1), "xr");

% Test data
ZT = X' * w1;
YT = 1./(1 + exp(-ZT));
testClasses = zeros(2, N);
testClasses(2,:) = cT;
for i = 1:N
    if YT(i, 1) >= 0.5
        testClasses(1, i) = 1;
    else
        testClasses(1, i) = 0;
    end
end

f4=figure('Name', "Test classes");
f4.Position = [100 100 1800 900];
hold on
xlim([-12 12])
ylim([-12 12])
title("Test classes")

plot(X(2, (testClasses(1, :) == 0)), X(3, (testClasses(1, :) == 0)), "ob");
plot(X(2, (testClasses(1, :) == 1)), X(3, testClasses(1, :) == 1), "or");

plot(X(2, (testClasses(2, :) == 0)), X(3, (testClasses(2, :) == 0)), "xb");
plot(X(2, (testClasses(2, :) == 1)), X(3, testClasses(2, :) == 1), "xr");

% Error rates
NbErrTrain = length(find(finalClasses(1,:) - finalClasses(2,:)));
trainRate = 100 * NbErrTrain / N;

NbErrTest = length(find(testClasses(1,:) - testClasses(2,:)));
testRate = 100 * NbErrTest / N;
