function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

for iter_x = 1 : m,
  layer1_data = Theta1 .* X(iter_x, :);
  layer1_sigm = sigmoid(sum(layer1_data, 2));

  layer1_sigm = [ones(1, 1); layer1_sigm];

  layer2_data = Theta2 * layer1_sigm;
  layer2_sigm = sigmoid(sum(layer2_data, 2));

  [value, index] = max(layer2_sigm);
  p(iter_x) = index;
end;

% =========================================================================

end
