#include <stdlib.h>
#include <matrix.h>
#include <test_utils.c>
#include <neuron.h>

#define ABS(X) ((X) >= 0 ? (X) : -(X))
#define CMP(A, B) (ABS(A - (B)) < 1e-15)

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
	shuffle_training_data(train);
	free_training_data(train);
}

void test_entrywise_prod()
{
	Matrix *m = create_matrix(2, 2);
	Matrix *n = create_matrix(2, 2);
	Matrix *x = create_matrix(2, 3);
	Matrix *tmp = create_matrix(2, 2);
	matrix_assign(tmp, 1.0, 4.0,
					   9.0, 16.0);

	matrix_assign(m, 1.0, 2.0,
				   3.0, 4.0);
	matrix_assign(n, 1.0, 2.0,
				   3.0, 4.0);

	Matrix *res = entrywise_product(m, x);
	ASSERT("Entrywise product of matrix with different shapes returns NULL", res == NULL);
	free_matrix(res);

	res = entrywise_product(m, n);
	ASSERT("Dummy entrywise product returns correct output.",
			matrix_cmp(res, tmp));

	free_matrix(m);
	free_matrix(n);
	free_matrix(x);
	free_matrix(tmp);
	free_matrix(res);
}

void test_matrix_addition()
{
	Matrix *a, *b, *c, *d;
	a = create_matrix(2, 2);
	b = create_matrix(2, 2);
	d = create_matrix(2, 2);
	c = create_matrix(2, 2);

	matrix_assign(a, 1.0, 2.0,
					 3.0, 4.0);
	matrix_assign(b, 1.0, 2.0,
					 3.0, 4.0);
	matrix_assign(c, 2.0, 4.0,
					 6.0, 8.0);

	d = matrix_add(a, b);
	ASSERT("Dummy matrix addition works.", matrix_cmp(c, d));
	free_matrix(a);
	free_matrix(b);
	free_matrix(c);
	free_matrix(d);
}

void test_matrix_to_array()
{
	double array[3] = {1.0, 2.0, 3.0};
	Matrix *m = create_matrix(3, 1);
	matrix_assign(m, 1.0, 2.0, 3.0);

	Matrix *x = array_to_matrix(array, 3);
	ASSERT("Array to matrix works.", matrix_cmp(x, m));

	matrix_assign(m, 1.1, 2.2, 3.3);
	matrix_to_array(m, array);
	ASSERT("Matrix to array works.",
			CMP(array[0], 1.10) && CMP(array[1], 2.20) && CMP(array[2], 3.30));
	free_matrix(m);
	free_matrix(x);
}

int main(int argc, char *argv[])
{
	test_shuffle_training();
	test_matrix_assign();
	test_matrix_prod();
	test_entrywise_prod();
	test_matrix_addition();
	test_matrix_to_array();
	return 0;
}
