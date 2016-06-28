#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "matrix.h"

#define SAME_SHAPE_CHECK(fn, operation, a, b) \
	if (a->n_rows != b->n_rows || a->n_cols != b->n_cols) { \
		fprintf(stderr, "%s ERROR: cannot compute the %s of a %dx%d matrix and a %dx%d matrix. They must have the same shapes.\n", fn, operation, a->n_rows, a->n_cols, b->n_rows, b->n_cols); \
		return NULL; \
	}

#define COPY_MATRIX_SHAPE(mat) create_matrix(mat->n_rows, mat->n_cols);
#define ABS(x) (((x) >= 0) ? (x) : -(x))
#define MATRIX_CMP_PREC 1e-8

/* Allocate memory for a matrix with n_rows rows and n_cols columns,
 * return a pointer to it. Must be freed with free_matrix(the_matrix)
 */
Matrix *create_matrix(int n_rows, int n_cols)
{
	int i;
	Matrix *mat = malloc(sizeof(Matrix));
	mat->n_rows = n_rows;
	mat->n_cols = n_cols;
	mat->data = malloc(sizeof(double *)*n_rows);
	for (i = 0; i < n_rows; i++) {
		mat->data[i] = malloc(sizeof(double)*n_cols);
	}
	fill_matrix(mat, 0);
	return mat;
}

/* Given a matrix and a double, fill all the matrix with this value. */
void fill_matrix(Matrix *mat, double value)
{
	int i, j;
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			mat->data[i][j] = value;
		}
	}
}

/* Free the memory allocated for a matrix */
void free_matrix(Matrix *mat)
{
	if (mat == NULL) {
		return;
	}
	int i;
	for (i = 0; i < mat->n_rows; i++) {
		free(mat->data[i]);
	}
	free(mat->data);
	free(mat);
}

/************ Matrix operations ************/

Matrix *matrix_prod(Matrix *a, Matrix *b)
{
	int i, j;
	int s;
	if (a->n_cols != b->n_rows) {
		fprintf(stderr, "matrix_prod ERROR: cannot multiply a %dx%d matrix and a %dx%d matrix.\n", a->n_rows, a->n_cols, b->n_rows, b->n_cols);
		return NULL;
	}
	Matrix *res = create_matrix(a->n_rows, b->n_cols);
	double val;
	for (i = 0; i < a->n_rows; i++) {
		for (j = 0; j < b->n_cols; j++) {
			val = 0.0;
			for (s = 0; s < a->n_cols; s++) {
				val += a->data[i][s] * b->data[s][j];
			}
			res->data[i][j] = val;
		}
	}
	return res;
}

/* Entrywise or Hadamardt product: produces another matrix where each
 * element ij is the product of elements ij of the original two
 * matrices.
 */
Matrix *entrywise_product(Matrix *a, Matrix *b)
{
	int i, j;
	Matrix *res = create_matrix(a->n_cols, b->n_cols);
	SAME_SHAPE_CHECK("entrywise_product", "entrywise product", a, b);
	for (i = 0; i < a->n_rows; i++) {
		for (j = 0; j < a->n_cols; j++) {
			res->data[i][j] = a->data[i][j] * b->data[i][j];
		}
	}
	return res;
}

/* Add two matrices */
Matrix *matrix_add(Matrix *a, Matrix *b)
{
	int i, j;
	SAME_SHAPE_CHECK("matrix_add", "matrix addition", a, b);
	Matrix *res = COPY_MATRIX_SHAPE(a);
	for (i = 0; i < a->n_rows; i++) {
		for (j = 0; j < a->n_cols; j++) {
			res->data[i][j] = a->data[i][j] + b->data[i][j];
		}
	}
	return res;
}

/* Tests if two matrices are equal (returns 1 if equal, else 0).
 * NOTE: if equal, they are equal to a precision MATRIX_CMP_PREC
 */
int matrix_cmp(Matrix *a, Matrix *b)
{
	if (a->n_rows != b->n_rows || a->n_cols != b->n_cols) {
		return 0;
	}
	int i, j;
	for (i = 0; i < a->n_rows; i++) {
		for (j = 0; j < a->n_cols; j++) {
			if (ABS(a->data[i][j] - b->data[i][j]) > MATRIX_CMP_PREC) {
				return 0;
			}
		}
	}
	return 1;
}

/* Assign values to a matrix:
 * eg. to create a matrix [[1, 2],
 *						   [3, 4]]:
 * matrix_assign(matrix, 1, 2,
 *						 3, 4);
 */
void matrix_assign(Matrix *mat, ...)
{
	va_list ap;
	double value;
	int rows = mat->n_rows;
	int cols = mat->n_cols;
	int i;
	va_start(ap, mat);
	for (i = 0; i < rows*cols; i++) {
		value = va_arg(ap, double);
		mat->data[i / cols][i % cols] = value;
		/* printf("value is %f\n", value); */
	}
	va_end(ap);
}

void matrix_print(Matrix *mat)
{
	int i, j;
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			printf("%f ", mat->data[i][j]);
						   /* j == (mat->n_cols - 1)? '\n': ' '); */
		}
		printf("\n");
	}
}

/********** End matrix operations **********/
