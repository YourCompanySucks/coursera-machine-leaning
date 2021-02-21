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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.3 Feedforward and cost functions()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a_1 = X ;

a_1 = [ones(m, 1) a_1];
z_2 = a_1 * Theta1' ;
a_2 = sigmoid(z_2) ;

a_2 = [ones(m, 1) a_2];
z_3 = a_2 * Theta2' ;
h = sigmoid(z_3) ;

y_k = y == 1:num_labels;

%[maxVal, p] = max(h, [], 2);

J = sum(sum((1/m) * ((-y_k) .* log(h) - (1 - y_k) .* log( 1 - h ))));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.4 regularized costFunction()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temp1 = Theta1;
temp1 = [zeros(size(temp1, 1), 1) temp1(:,2:end)];
temp2 = Theta2;
temp2 = [zeros(size(temp2, 1), 1) temp2(:,2:end)];

J = J + lambda / (2*m) * (sum(sum(temp1.^2)) + sum(sum(temp2.^2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 backpropagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% [step 1]
% 위에서 이미 구현함.

% [step 2]
delta_3 = h - y_k;

% [step 3]
delta_2 = delta_3 * Theta2 .* [ones(size(z_2, 1), 1) sigmoidGradient(z_2)];
delta_2 = delta_2(:,2:end);

% [step 4]
% 삼각형 뭐라고 하는지 모르겠네, 삼각형도 델타인거 같은데.. 변수 이름 뭐지 젠장 ㅋ
capital_delta_2 = delta_3' * a_2 ;
capital_delta_1 = delta_2' * a_1 ;

% [step 5]
Theta1_grad = (1/m) * capital_delta_1;
Theta2_grad = (1/m) * capital_delta_2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.5 regularized Neural Networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Theta1_grad =  Theta1_grad + lambda/m * temp1 ;
Theta2_grad =  Theta2_grad + lambda/m * temp2 ; 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
