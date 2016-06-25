#include <stdint.h>
#include "random.h"
#include <matrix.h>

#ifndef NEURON_H
#define NEURON_H

/* TrainData struct. This struct holds the data necessary to perform
 * the training and testing of a neural network. It must be freed with
 * free_training_data(the_ata);
 */
typedef struct {
	/* Number of training cases. */
	int n_train;
	/* Number of testing cases. */
	int n_test;
	int inputs_size;
	double **inputs_testing;;
	double *labels_testing;
	double **inputs_training;
	double *labels_training;
} TrainData;

/* Struct defining a neural network. Must be freed with
 * destroy_network(the_network);
 */
typedef struct network {
	/* number of layers */
	uint8_t n_layers;
	/* size of the layers */
	int *sizes;
	/* weights of the network: array of matrices */
	MatrixList weights;
	/* biases of the network */
	MatrixList biases;
} Network;

/*** Prototypes ***/

void free_training_data(TrainData *data);

TrainData *subset_training_data(TrainData *data, int start, int end);

void shuffle_training_data(TrainData *data);

Network *create_network(int n_layers, int *sizes);

void destroy_network(Network *net);

void SGD(Network *net, TrainData *data, int epochs,
	 int mini_batch_size, double learning_rate);

/*** End prototypes ***/

#endif // NEURON_H
