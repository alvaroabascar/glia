#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <byteswap.h>
#include <neuron.h>

/* Prototypes */
TrainData *mnist_load(char *path);
void *concat(char *str1, char *str2);

/* Load MNIST labels, given the file path. Return them as an "array"
 * of uint8
 */
double **load_labels(char *path, int *n_items)
{
	int i, j;
	int32_t magic;
	char *labels;
	double **labels_double;
	FILE *stream = fopen(path, "r");
	if (!stream) {
		fprintf(stderr, "Could not load file %s. Aborting :(.\n", path);
		return NULL;
	}
	fread(&magic, 4, 1, stream);
	/* Read number of items */
	fread(n_items, 4, 1, stream);
	if (magic != 2049) {
		*n_items = __bswap_32(*n_items);
	}
	labels = malloc(*n_items);
	fread(labels, 1, *n_items, stream);
	fclose(stream);
	labels_double = malloc(sizeof(double *)*(*n_items));
	for (i = 0; i < *n_items; i++) {
		labels_double[i] = malloc(sizeof(double) * 10);
		for (j = 0; j < 10; j++) {
			labels_double[i][j] = 0.0;
		}
		labels_double[i][labels[i]] = 1.0;
	}
	free(labels);
	return labels_double;
}

/* Load the MNIST images, given the file path. Return them as an
 * "array of matrices".
 */
double **load_images(char *path, int *n_items, int *n_rows, int *n_cols)
{
	int i, item;
	int32_t magic;
	double **inputs;
	FILE *stream = fopen(path, "r");
	uint8_t *tmp_uint8;
	if (!stream) {
		fprintf(stderr, "Could not load file %s. Aborting :(.\n", path);
		return NULL;
	}
	fread(&magic, 4, 1, stream);
	fread(n_items, 4, 1, stream);
	fread(n_rows, 4, 1, stream);
	fread(n_cols, 4, 1, stream);
	if (magic != 2051) {
		*n_items = __bswap_32(*n_items);
		*n_rows = __bswap_32(*n_rows);
		*n_cols = __bswap_32(*n_cols);
	}
	inputs = malloc(sizeof(void **) * (*n_items));
	tmp_uint8 = malloc((*n_cols) * (*n_rows));
	for (item = 0; item < *n_items; item++) {
		inputs[item] = malloc((*n_cols) * (*n_rows) * sizeof(double));
		fread(tmp_uint8, 1, (*n_cols) * (*n_rows), stream);
		for (i = 0; i < (*n_cols) * (*n_rows); i++) {
			inputs[item][i] = (double)tmp_uint8[i] / 255.0;
		}
	}
	free(tmp_uint8);
	fclose(stream);
	return inputs;
}

/* Load the mnist dataset given the folder path & result a struct with
 * all the data.
 */
TrainData *mnist_load(char *path)
{
	int32_t n_train, n_test, n_rows, n_cols;
	double **labels_train;
	double **labels_test;
	double **images_train;
	double **images_test;
	char *train_images_path = concat(path, "/train-images-idx3-ubyte");
	char *train_labels_path = concat(path, "/train-labels-idx1-ubyte");
	char *test_labels_path = concat(path, "/t10k-labels-idx1-ubyte");
	char *test_images_path = concat(path, "/t10k-images-idx3-ubyte");
	TrainData *data = malloc(sizeof(TrainData));
	labels_train = load_labels(train_labels_path, &n_train);
	fprintf(stderr, "Loaded %s: %d labels.\n", train_labels_path, n_train);

	labels_test = load_labels(test_labels_path, &n_test);
	fprintf(stderr, "Loaded %s: %d labels.\n", test_labels_path, n_train);

	images_train = load_images(train_images_path, &n_train, &n_rows, &n_cols);
	fprintf(stderr, "Loaded %s: %d %dx%d images.\n",
			train_images_path, n_train, n_rows, n_cols);

	images_test = load_images(test_images_path, &n_test, &n_rows, &n_cols);
	fprintf(stderr, "Loaded %s: %d %dx%d images.\n",
			train_images_path, n_train, n_rows, n_cols);

	data->n_train = n_train;
	data->n_test = n_test;
	data->inputs_size = n_rows * n_cols;
	data->outputs_size = 10;
	data->labels_testing = labels_test;
	data->labels_training = labels_train;
	data->inputs_testing = images_test;
	data->inputs_training = images_train;

	/* Free all */
	free(train_images_path);
	free(train_labels_path);
	free(test_images_path);
	free(test_labels_path);
	
	return data;
}

/* Concatenate two strings & return the result as a new one */
void *concat(char *str1, char *str2)
{
	void *r = malloc(strlen(str1) + strlen(str2) + 1);
	strcpy(r, str1);
	r = strcat(r, str2);
	return r;
}

/* Load the MNIST dataset, create & train a network */
int main(int argc, char *argv[])
{
	if (argc != 2) {
		exit(1);
	}
	char *path = argv[1];
	TrainData *data = mnist_load(path);
	fprintf(stderr, "Loading completed.\n");

	Network *net = create_network(3, 768, 30, 10);
	fprintf(stderr, "Network created.\n");

	/* matrix_print(array_to_matrix(data->inputs_training[0], 768)); */
	fprintf(stderr, "Initial acc: %f\n", test_accuracy(net, data));
	SGD(net, data, 30, 10, 3.0);
	fprintf(stderr, "SGD completed.\n");
	free_training_data(data);
	fprintf(stderr, "Data freed.\n");
	destroy_network(net);
	fprintf(stderr, "Network destructed.\n");
}
