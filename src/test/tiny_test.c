#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <byteswap.h>
#include <neuron.h>

#define DEBUG(mat) matrix_print_shape(mat); matrix_print(mat);

int main()
{
	TrainData *data = malloc(sizeof(TrainData));
	data->n_train = 1;
	data->n_test = 0;
	double **labels_training = malloc(sizeof(double *) * 1);
	double **inputs_training = malloc(sizeof(double *) * 1);

	inputs_training[0] = malloc(sizeof(double) * 2);
	inputs_training[0][0] = 1.0;
	inputs_training[0][1] = 1.0;

	labels_training[0] = malloc(sizeof(double) * 1);
	labels_training[0][0] = 1.0;
	data->inputs_training = inputs_training;
	data->labels_training = labels_training;

	Network *net = create_network(3, 2, 2, 1);
	fprintf(stderr, "%d %d %d\n", net->sizes[0], net->sizes[1], net->sizes[2]);
	SGD(net, data, 1, 1, 3.0, 0.0);
	DEBUG(net->weights[1]);
	destroy_network(net);
	free_training_data(data);
}
