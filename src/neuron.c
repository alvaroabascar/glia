#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>

#include <utils.h>
#include <neuron.h>
#include <matrix.h>
#include <random.h>

#define DEBUG(mat) matrix_print_shape(mat); matrix_print(mat);

/*
 *
 * A simple Neural Network library.
 *
 */

/* Free the memory allocated for a TrainData struct. */
void free_training_data(TrainData *data)
{
	int i;
	for (i = 0; i < data->n_train; i++) {
		free(data->inputs_training[i]);
		free(data->labels_training[i]);
	}
	for (i = 0; i < data->n_test; i++) {
		free(data->inputs_testing[i]);
		free(data->labels_testing[i]);
	}
	free(data->inputs_training);
	free(data->inputs_testing);
	free(data->labels_testing);
	free(data->labels_training);
	free(data);
}

/* Shuffle training data (inputs & labels) using the Fisher & Yates
 * algorithm. Don't touch the testing data.
 */
void shuffle_training_data(TrainData *data)
{
    int i;
    long j;
	double *tmp_label, *tmp_inputs;
	srand(time(NULL));
    for (i = 0; i < data->n_train - 1; i++) {
	    j = random_in_range(i, data->n_train - 1);
		/* Exchange labels[i] and labels[j] */
		tmp_label = data->labels_training[i];
		data->labels_training[i] = data->labels_training[j];
		data->labels_training[j] = tmp_label;
		/* Exchange inputs[i] and inputs[j] */
		tmp_inputs = data->inputs_training[i];
		data->inputs_training[i] = data->inputs_training[j];
		data->inputs_training[j] = tmp_inputs;
    }
}

/* Given a struct TrainData, a 'start' index and a number of items 'n',
 * return a subset of 'n' consecutive items, starting from 'start'.
 *
 * NOTE: subset is done on the training data, the testing data is
 * kept untouched.
 *
 * NOTE: the items inside the new TrainData POINT TO items in the old
 * one. Thus the new structured must be freed with free(subset_data)
 * and the original struct must be freed with free_training_data(data).
 * DO NOT free the new struct calling free_training_data(subset_struct)
 */
TrainData *subset_training_data(TrainData *data, int start, int n)
{
	int i;
	TrainData *ndata = malloc(sizeof(TrainData));
	ndata->n_train = n;
	ndata->n_test = data->n_test;
	ndata->inputs_size = data->inputs_size;
	// Load testing
	ndata->inputs_testing = data->inputs_testing;
	ndata->labels_testing = data->labels_testing;
	ndata->inputs_training = data->inputs_training + start;
	ndata->labels_training = data->labels_training + start;
	return ndata;
}

/* Initialize & return a pointer to a new network:
 * n_layers: number of layers of the net, including input and output.
 * sizes: array of int. sizes[i] indicates the number of neurons in
 *		  the ith layer.
 */
Network *create_network(int n_layers, ...)
{
	int i, sizes[n_layers];
	va_list ap;
	va_start(ap, n_layers);
	for (i = 0; i < n_layers; i++) {
		sizes[i] = va_arg(ap, int);
	}
	va_end(ap);
	Network *net = malloc(sizeof(Network));
	net->n_layers = n_layers;

	net->sizes = malloc(sizeof(int) * n_layers);
	net->weights = malloc(sizeof(Matrix *)*(n_layers - 1));
	net->biases = malloc(sizeof(Matrix *)*(n_layers - 1));

	arrncpy(net->sizes, sizes, n_layers);

	for (i = 0; i < n_layers - 1; i++) {
		net->weights[i] = create_matrix(sizes[i+1], sizes[i]);
		/* Fill it with a gaussian (mean 0 variance 1) */
		matrix_fill_gaussian_random(net->weights[i]);
		/* matrix_fill(net->weights[i], 0.1); */
		net->biases[i] = create_matrix(sizes[i+1], 1);
		/* Fill it with a gaussian (mean 0 variance 1) */
		/* matrix_fill(net->biases[i], 0.1); */
		matrix_fill_gaussian_random(net->biases[i]);
	}
	return net;
}

/* Free the memory assigned to a network. */
void destroy_network(Network *net)
{
	if (net == NULL) {
		return;
	}
	int i;
	for (i = 0; i < (net->n_layers)-1; i++) {
		free_matrix(net->weights[i]);
		free_matrix(net->biases[i]);
	}
	free(net->weights);
	free(net->biases);
	free(net->sizes);
	free(net);
}

/* Perform stochastic gradient descent. */
void SGD(Network *net, TrainData *data, int n_epochs,
		 int mini_batch_size, double learning_rate, double lambda)
{

	int epoch, start, batch;
    int	n_mini_batches = data->n_train / mini_batch_size;
	double acc;
	TrainData *mini_batch;
	/* Loop through each epoch */
	for (epoch = 0; epoch < n_epochs; epoch++) {
		shuffle_training_data(data);
		for (batch = 0; batch < n_mini_batches; batch++) {
			start = batch * mini_batch_size;
			mini_batch = subset_training_data(data, start,
			                                  mini_batch_size);
			network_update_mini_batch(net, mini_batch, learning_rate,
									  lambda, data->n_train);
			free(mini_batch);
		}
		fprintf(stderr, "Epoch %d finished.\n", epoch);
		acc = test_accuracy(net, data);
		fprintf(stderr, "Accuracy: %.2f%%\n", acc * 100);
	}
}

void network_update_mini_batch(Network *net, TrainData *mini_batch,
							   double learning_rate, double lambda,
							   int N_total)
{
	/* 
	 * Update the weights and biases of the network using a mini batch of
	 * N elements of the training set.
	 *
	 * **********************************************************************
	 * INPUTS:
	 *
	 * net -> the network.
	 * mini_batch -> a subset of the training set.
	 * learning_rate -> the learning rate (eta).
	 * lambda -> the regularization parameter (for L2 regularization).
	 * **********************************************************************
	 *
	 * Explanation of the procedure:
	 * ----------------------------
	 *
	 * Backpropagation is performed for each input of the batch, obtaining the
	 * gradients for the weights and biases. All the gradients are summed
	 * together in nabla_weights and nabla_biases.
	 *
	 * When lambda = 0 (no L2 regularization) the weights and biases are
	 * updated according to the formula:
	 *
	 * W = W - (eta/N)*(nabla_weights)
	 * B = B - (eta/N)*(nabla_biases)
	 *
	 * - N = number of items in the training set
	 * - eta = learning rate
	 * - W = weights
	 * - B = biases
	 * - nabla_weights, nabla_biases = sum of gradients with respect to weights
	 * and biases.
	 *
	 * When lambda != 0, we have L2 regularization. The formula for the biases
	 * remains the same, but for the weights it becomes:
	 *
	 * W = W - (eta/N_total)*(nabla_weights) - (eta*lambda/N)*W
	 *
	 * Here, N_total = total number of items in the training set
	 *
	 * This is equal to:
	 *
	 * W = (1 - eta*lambda/N_total)*W - (eta/N)*(nabla_weights)
	 *
	 * In the current implementation, we first modify W according to the first
	 * term, then update it normally using the gradients.
	 *
	 */
	int i, j;
	double eta_over_n, l2_term;
	MatrixList nabla_weights, nabla_biases; // Cumulative gradients.
	MatrixList delta_weights, delta_biases; // Temporal gradients.

	nabla_weights = malloc(sizeof(Matrix *) * (net->n_layers - 1));
	nabla_biases = malloc(sizeof(Matrix *) * (net->n_layers - 1));
	delta_weights = malloc(sizeof(Matrix *) * (net->n_layers - 1));
	delta_biases = malloc(sizeof(Matrix *) * (net->n_layers - 1));

	/* Initialize gradient of weights and biases as zero. */
	for (i = 0; i < net->n_layers - 1; i++) {
		nabla_weights[i] = create_matrix_zeros(net->sizes[i+1],
										 net->sizes[i]);
		nabla_biases[i] = create_matrix_zeros(net->sizes[i+1], 1);
	}
	/* backpropagate, calculate gradient for each training input & add
	 * it to the cumulative gradient.
	 */
	for (i = 0; i < mini_batch->n_train; i++) {
		/* 1. Calculate the gradient for a single input */
		backpropagate(net, mini_batch->inputs_training[i],
						   mini_batch->labels_training[i],
						   delta_weights,
						   delta_biases);
		/* 2. Add it to the cumulative gradients. */
		for (j = 0; j < net->n_layers - 1; j++) {
			matrix_add(nabla_weights[j], delta_weights[j]);
			matrix_add(nabla_biases[j], delta_biases[j]);

			free_matrix(delta_weights[j]);
			free_matrix(delta_biases[j]);
		}
	}
	/* Update weights with the formula:
	 * W = (1 - eta*lambda/N_TOTAL)*W - (eta/N)*(nabla_weights) */

	l2_term = (1 - learning_rate * lambda / (double)N_total);
	eta_over_n = -learning_rate / (double)(mini_batch->n_train);
	for (j = 0; j < net->n_layers - 1; j++) {
		/* First: apply the first term: W = W * (1 - eta*lambda/N_TOTAL) */
		matrix_multiply(net->weights[j], l2_term); 
		/* Second: substract the gradient (add -(eta/N)*nabla_weight) */
		matrix_multiply(nabla_weights[j], eta_over_n);
		matrix_add(net->weights[j], nabla_weights[j]);

		matrix_multiply(nabla_biases[j], eta_over_n);
		matrix_add(net->biases[j], nabla_biases[j]);
	}
	for (i = 0; i < net->n_layers - 1; i++) {
		free_matrix(nabla_weights[i]);
		free_matrix(nabla_biases[i]);
	}
	free(nabla_weights);
	free(nabla_biases);
	free(delta_weights);
	free(delta_biases);
}

/* Set the inputs of the network and propagate until getting the output. */
Matrix *feedforward(Network *net, double *input)
{
	Matrix *as = array_to_matrix(input, net->sizes[0]);
	Matrix *zs;
	int i;
	for (i = 0; i < net->n_layers - 1; i++) {
		zs = matrix_prod_optim(net->weights[i], as);
		matrix_add(zs, net->biases[i]);
		as = sigmoid_vect(zs);
		free_matrix(zs);
	}
	return as;
}

void backpropagate(Network *net, double *inputs, double *outputs,
				   MatrixList delta_weights, MatrixList delta_biases)
{
	int i;
	Matrix *errors, *errors_new, *weights_T,
		   *sigma_prime, *as_T, *outs;
	/* Feedforward pass */
	MatrixList zs = malloc(sizeof(Matrix *)*net->n_layers);
	MatrixList as = malloc(sizeof(Matrix *)*net->n_layers);
	as[0] = array_to_matrix(inputs, net->sizes[0]);
	zs[0] = create_matrix(1, 1); // unused
	for (i = 0; i < net->n_layers - 1; i++) {
		zs[i+1] = matrix_prod_optim(net->weights[i], as[i]);
		matrix_add(zs[i+1], net->biases[i]);
		as[i+1] = sigmoid_vect(zs[i+1]);
	}
	/* Calculate errors in last layer */
	outs = array_to_matrix(outputs, net->sizes[net->n_layers-1]);
	errors = cost_derivative(outs, as[net->n_layers-1]);

    delta_biases[net->n_layers-2] = matrix_copy(errors);
	as_T = transpose(as[net->n_layers-2]);
    delta_weights[net->n_layers-2] = matrix_prod_optim(errors, as_T);

	free_matrix(outs);
	/* Backpropagate */
	for (i = net->n_layers - 3; i >= 0; i--) {
		weights_T = transpose(net->weights[i+1]);
		/* Errors in current layer */
		errors_new = matrix_prod_optim(weights_T, errors);
		sigma_prime = sigmoid_prime_vect(zs[i+1]);

		matrix_entrywise_product(errors_new, sigma_prime); 
		as_T = transpose(as[i]);

		delta_weights[i] = matrix_prod_optim(errors_new, as_T);
		delta_biases[i] = matrix_copy(errors_new);

		free_matrix(errors);
		free_matrix(weights_T);
		free_matrix(as_T);
		free_matrix(sigma_prime);

		errors = errors_new;
	}
	free_matrix(errors);
	for (i = 0; i < net->n_layers; i++) {
		free_matrix(zs[i]);
		free_matrix(as[i]);
	}
	free(zs);
	free(as);
}

/* Sigmoid function */
double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

/* Derivative of the sigmoid function */
double sigmoid_prime(double x)
{
	return sigmoid(x) * (1.0 - sigmoid(x));
}

/* Vectorized version of the sigmoid function */
Matrix *sigmoid_vect(Matrix *mat)
{
	int i, j;
	Matrix *newmat = create_matrix(mat->n_rows, mat->n_cols);
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			newmat->data[i][j] = sigmoid(mat->data[i][j]);
		}
	}
	return newmat;
}

/* Vectorized version of the derivative of the sigmoid function */
Matrix *sigmoid_prime_vect(Matrix *mat)
{
	int i, j;
	Matrix *newmat = create_matrix(mat->n_rows, mat->n_cols);
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			newmat->data[i][j] = sigmoid_prime(mat->data[i][j]);
		}
	}
	return newmat;
}

Matrix *sigmoid_prime_from_sigmoid_vect(Matrix *mat)
{
	int i, j;
	Matrix *newmat = create_matrix(mat->n_rows, mat->n_cols);
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			newmat->data[i][j] = mat->data[i][j]*(1 - mat->data[i][j]);
		}
	}
	return newmat;
}

double test_accuracy(Network *net, TrainData *data)
{
	int i;
	double *input, *output, *correct_output;
	Matrix *out_mat;
	int n_ok = 0;
	int s = data->outputs_size;
	output = malloc(sizeof(double) * s);
	for (i = 0; i < data->n_test; i++) {
		input = data->inputs_testing[i];
		correct_output = data->labels_testing[i];

		out_mat = feedforward(net, input);
		matrix_to_array(out_mat, output);
		free_matrix(out_mat);

		if (argmax(output, s) == argmax(correct_output, s)) {
			n_ok += 1;
		}
	}
	return ((double)n_ok) / ((double)(data->n_test));
}

/*
 * Compute the errors in the last layer, given the correct outputs and
 * the activations from the last layer as matrices.
 * Returns the cross-entropy derivative.
 */
Matrix *cost_derivative(Matrix *outputs, Matrix *activs)
{
	Matrix *errs = create_matrix(outputs->n_rows, outputs->n_cols);
	matrix_add(errs, activs);
	matrix_substract(errs, outputs);
	return errs;
}
