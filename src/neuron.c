#include <stdio.h>
#include <stdlib.h>
#include "neuron.h"

/*
 *
 * A simple Neural Network library.
 *
 */

void free_training_data(train_data data)
{
	int i;
	for (i = 0; i < data.n_train; i++) {
		free(data.images_training[i]);
	}
	for (i = 0; i < data.n_test; i++) {
		free(data.images_testing[i]);
	}
	free(data.images_training);
	free(data.images_testing);
	free(data.labels_testing);
	free(data.labels_training);
}
