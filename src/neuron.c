#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <utils.h>
#include <neuron.h>
#include <matrix.h>
#include <random.h>

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
	}
	for (i = 0; i < data->n_test; i++) {
		free(data->inputs_testing[i]);
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
	double tmp_label, *tmp_inputs;
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
Network *create_network(int n_layers, int *sizes)
{
	int i;
	Network *net = malloc(sizeof(Network));
	net->n_layers = n_layers;

	net->sizes = malloc(sizeof(int) * n_layers);
	net->weights = malloc(sizeof(Matrix *)*(n_layers - 1));
	net->biases = malloc(sizeof(Matrix *)*(n_layers - 1));

	arrncpy(net->sizes, sizes, n_layers);

	for (i = 0; i < n_layers - 1; i++) {
		net->weights[i] = create_matrix(sizes[i+1], sizes[i]);
		net->biases[i] = create_matrix(sizes[i+1], 1);
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
		 int mini_batch_size, double learning_rate)
{

	int epoch, start, batch;
    int	n_mini_batches = data->n_train / mini_batch_size;
	TrainData *mini_batch;
	/* Loop through each epoch */
	for (epoch = 0; epoch < n_epochs; epoch++) {
		/* shuffle_training_data(data); */
		for (batch = 0; batch < n_mini_batches; batch++) {
			start = batch * mini_batch_size;
			mini_batch = subset_training_data(data, start,
			                                  mini_batch_size);
			free(mini_batch);
		}
	}
}

/* Sigmoid function */
double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

/* Derivative of the sigmoid function */
double sigmoid_prime(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
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
