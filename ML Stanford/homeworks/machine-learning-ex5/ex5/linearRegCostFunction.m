function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

diff_value = (X * theta - y);
theta_no_first = theta(2 : end);

J = sum(diff_value(:) .^ 2) / (2 * m) + sum(theta_no_first(:) .^ 2) * lambda / (2 * m);

diff_value_mul_1 = diff_value .* X(: , 1);
diff_value_mul_2 = diff_value .* X(: , 2 : end);

diff_value_mul = diff_value .* X;
grad = sum(diff_value_mul(: , :)) / m ;

for it = 2 : size(grad(:)),
  grad(it) += lambda * theta(it) / m;
end;

% =========================================================================

grad = grad(:);

end