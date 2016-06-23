#include <stdint.h>
#include <matrix.h>

#ifndef NEURON_H
#define NEURON_H

typedef struct {
	int n_train;
	int n_test;
	int n_rows;
	int n_cols;
	uint8_t **images_testing;;
	uint8_t *labels_testing;
	uint8_t *labels_training;
	uint8_t **images_training;
} TrainData;

typedef struct network {
	uint8_t n_layers;	 /* number of layers */
	int *sizes;		 /* size of the layers */
	MatrixList weights;	 /* weights of the network: array of matrices */
	MatrixList biases;       /* biases of the network */
} Network;

void free_training_data(TrainData data);

Network *create_network(int n_layers, int *sizes);
void destroy_network(Network *net);

#endif // NEURON_H
