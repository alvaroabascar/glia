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
	int outputs_size;
	double **inputs_testing;;
	double **labels_testing;
	double **inputs_training;
	double **labels_training;
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

Network *create_network(int n_layers, ...);

void destroy_network(Network *net);

void SGD(Network *net, TrainData *data, int epochs,
	 int mini_batch_size, double learning_rate, double lambda);

void network_update_mini_batch(Network *net,
		TrainData *mini_batch, double learning_rate, double lambda,
		int N_total);

Matrix *feedforward(Network *net, double *input);

void backpropagate(Network *net, double *inputs, double *outputs,
				   MatrixList delta_weigths, MatrixList delta_biases);

double sigmoid(double x);
double sigmoid_prime(double x);
Matrix *sigmoid_vect(Matrix *mat);
Matrix *sigmoid_prime_vect(Matrix *mat);
Matrix *sigmoid_prime_from_sigmoid_vect(Matrix *mat);
Matrix *cost_derivative(Matrix *outputs, Matrix *activs);
double test_accuracy(Network *net, TrainData *data);

/*** End prototypes ***/

#endif // NEURON_H
