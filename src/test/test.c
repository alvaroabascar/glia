#include <stdlib.h>
#include <matrix.h>
#include <test_utils.c>
#include <neuron.h>

int test_matrix_prod()
{
	printf("\n** BLOCK matrix_prod **\n");

	Matrix *mat1 = create_matrix(3, 2);
	Matrix *mat2 = create_matrix(3, 2);
	Matrix *mat3 = create_matrix(2, 3);
	Matrix *res = matrix_prod(mat1, mat2);

	ASSERT("(n x m) * (a x b) returns NULL if m != a ",
		   res == NULL);

	matrix_assign(mat1, 1.0, 2.0,
						3.0, 4.0,
						5.0, 6.0);

	matrix_assign(mat3, 2.0, 3.0, 4.0,
						5.0, 6.0, 7.0);

	free_matrix(res);
	res = create_matrix(3, 3);

	matrix_assign(res, 12.0, 15.0, 18.0,
				       26.0, 33.0, 40.0,
					   40.0, 51.0, 62.0);
	Matrix *result = matrix_prod(mat1, mat3);
	ASSERT("dummy product is fine",
           matrix_cmp(res, result));

	free_matrix(mat1);
	free_matrix(mat2);
	free_matrix(mat3);
	free_matrix(res);
	free_matrix(result);
}

int test_matrix_assign()
{
	printf("\n** BLOCK matrix_assign **\n");

	Matrix *mat1 = create_matrix(2, 2);
	Matrix *mat2 = create_matrix(2, 2);
	mat1->data[0][0] = 1.0;
	mat1->data[0][1] = 2.0;
	mat1->data[1][0] = 3.0;
	mat1->data[1][1] = 4.0;
	matrix_assign(mat2, 1.0, 2.0,
						3.0, 4.0);
	ASSERT("matrix_assign works fine.",
			matrix_cmp(mat1, mat2));
	free_matrix(mat1);
	free_matrix(mat2);
}

void test_shuffle_training()
{
	int i;
	TrainData *train = malloc(sizeof(TrainData));
	train->n_train = 3;
	train->inputs_training = malloc(sizeof(double *)*3);
	train->labels_training = malloc(sizeof(double)*3);
	for (i = 0; i < 3; i++) {
	    train->inputs_training[i] = malloc(sizeof(double)*3);
	    train->inputs_training[i][0] = (double)i;
	    train->inputs_training[i][0] = (double)i;
	    train->inputs_training[i][0] = (double)i;
	    train->labels_training[i] = (double)i;
	}
	train->n_test = 0;
	for (i = 0; i < 3; i++) {
		printf("%d -> %f\n", i, train->labels_training[i]);
	}
	shuffle_training_data(train);
	for (i = 0; i < 3; i++) {
		printf("%d -> %f\n", i, train->labels_training[i]);
	}
	free_training_data(train);
}

int main(int argc, char *argv[])
{
	test_matrix_assign();
	test_matrix_prod();
	test_shuffle_training();
	return 0;
}
