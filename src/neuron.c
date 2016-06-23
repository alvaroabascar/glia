#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <neuron.h>
#include <matrix.h>

/*
 *
 * A simple Neural Network library.
 *
 */

/* Free the memory allocated for a TrainData struct. */
void free_training_data(TrainData data)
{
	int i;
	for (i = 0; i < data.n_train; i++) {
		free(data.inputs_training[i]);
	}
	for (i = 0; i < data.n_test; i++) {
		free(data.inputs_testing[i]);
	}
	free(data.inputs_training);
	free(data.inputs_testing);
	free(data.labels_testing);
	free(data.labels_training);
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
