function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
% J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%size_theta = size(theta, 1);
% vct_x = zeros(size(X, 1), 1);
% for iter = 1 : size_theta,
%   vct_x = vct_x + X(: , iter) * theta(iter);
% end;

% vct_res = (vct_x - y) .^ 2;
% J = sum(vct_res(:)) / (2 * m);

vct_x = X * theta;
vct_diff = vct_x - y;
J = (vct_diff' * vct_diff) / (2 * m);

% =========================================================================

end
