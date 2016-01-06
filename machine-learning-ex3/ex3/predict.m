function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

% Add bias unit to the input vectors
X = [ones(m, 1) X];

%Theta1 is the vector which describes weights for input layer -> hidden layer
%Theta2 is the vector which described weights for hidden layer -> output layer
hiddenLayerActivation = sigmoid(X * Theta1');

% add bias unit to hidden layer activation vector
hiddenLayerActivation = [ones(m, 1) hiddenLayerActivation];

[temp, p] = max( sigmoid( hiddenLayerActivation * Theta2'), [], 2);

end
