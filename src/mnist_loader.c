#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <byteswap.h>
#include <neuron.h>

/* Prototypes */
train_data mnist_load(char *path);
void *concat(char *str1, char *str2);

uint8_t *load_labels(char *path, int *n_items)
{
	int32_t magic;
	void *labels;
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
	return labels;
}

uint8_t **load_images(char *path, int *n_items, int *n_rows, int *n_cols)
{
	int i, item;
	int32_t magic;
	uint8_t **images;
	FILE *stream = fopen(path, "r");
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
	images = malloc(sizeof(void **) * (*n_items));
	for (i = 0; i < *n_items; i++) {
		images[i] = malloc((*n_cols) * (*n_rows));
		fread(images[i], 1, (*n_cols) * (*n_rows), stream);
	}
	fclose(stream);
	return images;
}

train_data mnist_load(char *path)
{
	int32_t n_train, n_test, n_rows, n_cols;
	uint8_t *labels_train;
	uint8_t *labels_test;
	uint8_t **images_train;
	uint8_t **images_test;
	char *train_images_path = concat(path, "/train-images-idx3-ubyte");
	char *train_labels_path = concat(path, "/train-labels-idx1-ubyte");
	char *test_labels_path = concat(path, "/t10k-labels-idx1-ubyte");
	char *test_images_path = concat(path, "/t10k-images-idx3-ubyte");
	train_data data;
	labels_train = load_labels(train_labels_path, &n_train);
	fprintf(stdout, "Loaded %s: %d labels.\n", train_labels_path, n_train);

	labels_test = load_labels(test_labels_path, &n_test);
	fprintf(stdout, "Loaded %s: %d labels.\n", test_labels_path, n_train);

	images_train = load_images(train_images_path, &n_train, &n_rows, &n_cols);
	fprintf(stdout, "Loaded %s: %d %dx%d images.\n",
			train_images_path, n_train, n_rows, n_cols);

	images_test = load_images(test_images_path, &n_test, &n_rows, &n_cols);
	fprintf(stdout, "Loaded %s: %d %dx%d images.\n",
			train_images_path, n_train, n_rows, n_cols);

	data.n_train = n_train;
	data.n_test = n_test;
	data.n_cols = n_rows;
	data.n_rows = n_cols;
	data.labels_testing = labels_test;
	data.labels_training = labels_train;
	data.images_testing = images_test;
	data.images_training = images_train;

	/* Free all */
	free(train_images_path);
	free(train_labels_path);
	free(test_images_path);
	free(test_labels_path);
	
	return data;
}

void *concat(char *str1, char *str2)
{
	void *r = malloc(strlen(str1) + strlen(str2) + 1);
	strcpy(r, str1);
	r = strcat(r, str2);
	return r;
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		exit(1);
	}
	char *path = argv[1];
	train_data data = mnist_load(path);
	free_training_data(data);
}

