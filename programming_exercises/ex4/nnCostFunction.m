function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Cost function calculation
a1 = [ones(size(X, 1), 1) X];

z2 = a1 * Theta1';
g2 = sigmoid(z2);
a2 = [ones(size(z2, 1), 1) g2];

z3 = a2 * Theta2';
h_x = a3 = g3 = sigmoid(z3);

y_onehot = y == 1:num_labels;

J_inner_calc = (-y_onehot .* log(h_x)) - ((1 - y_onehot) .* log(1 - h_x));
J_sum_for_k = sum(J_inner_calc);
J_sum_for_m = sum(J_sum_for_k);
cost_function = J_sum_for_m / m;


% Regularization
reg_theta1 = Theta1;
reg_theta1(:, 1) = 0;
reg_theta1_term = sum(sum(reg_theta1.^2));

reg_theta2 = Theta2;
reg_theta2(:, 1) = 0;
reg_theta2_term = sum(sum(reg_theta2.^2));
reg_term = lambda * (reg_theta1_term + reg_theta2_term) / (2 * m);


% Backpropagation (Delta term)
delta3 = a3 - y_onehot;
g2_prm = sigmoidGradient(z2);
g2_prm_w_b = [ones(size(g2_prm, 1), 1) g2_prm];
delta2 = (delta3 * Theta2) .* g2_prm_w_b;
delta2_only_val = delta2(:,2:end);

Theta2_grad = (delta3' * a2)' / m;
Theta1_grad = (delta2_only_val' * a1) / m;


% J defenition
J = cost_function + reg_term


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
