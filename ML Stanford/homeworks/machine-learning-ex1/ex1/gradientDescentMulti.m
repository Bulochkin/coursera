function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    temp = theta;

    vct_x = X * theta;
    vct_diff = (vct_x - y);

    for iter2 = 1 : size(theta, 1),
      vct_res = vct_diff .* X(: , iter2);
      vct_sum = alpha * sum(vct_res(:)) / m;
      temp(iter2, 1) = theta(iter2, 1) - vct_sum;
    end;

    theta = temp;

    % vec_X = X * theta;
    % vec_Diff = vec_X - y;
    %
    % vec_X1 = vec_Diff .* X(: , 1);
    % vec_Sum1 = alpha * sum(vec_X1(:)) / m;
    %
    % vec_X2 = vec_Diff .* X(: , 2);
    % vec_Sum2 = alpha * sum(vec_X2(:)) / m;
    %
    % vec_X3 = vec_Diff .* X(: , 3);
    % vec_Sum3 = alpha * sum(vec_X3(:)) / m;
    %
    % temp1 = theta(1) - vec_Sum1;
    % temp2 = theta(2) - vec_Sum2;
    % temp3 = theta(3) - vec_Sum3;
    %
    % theta(1) = temp1;
    % theta(2) = temp2;
    % theta(3) = temp3;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
