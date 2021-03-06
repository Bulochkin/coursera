function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
% J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

vct_x = theta(1) + X(: , 2) * theta(2);
vct_res = (vct_x - y) .^ 2;
J = sum(vct_res(:)) / (2 * m);

% sum = 0;
% for iter = 1 : m ,
%   h_0 = theta(1) + theta(2) * X(iter, 2);
%   sum = sum + (h_0 - y(iter)) ^ 2;
% end;
%
% J = sum / (2 * m);

% =========================================================================

end
