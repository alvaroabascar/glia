#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

#include <random.h>
#include <matrix.h>

#define SAME_SHAPE_CHECK(fn, operation, a, b, rval) \
	if (a->n_rows != b->n_rows || a->n_cols != b->n_cols) { \
		fprintf(stderr, "%s ERROR: cannot compute the %s of a %dx%d matrix and a %dx%d matrix. They must have the same shapes.\n", fn, operation, a->n_rows, a->n_cols, b->n_rows, b->n_cols); \
		return rval; \
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
	matrix_fill(mat, 0);
	return mat;
}

/* Given a matrix and a double, fill all the matrix with this value. */
void matrix_fill(Matrix *mat, double value)
{
	int i, j;
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			mat->data[i][j] = value;
		}
	}
}

/* Fill with uniform randoms between 0 and 1 */
void matrix_fill_random(Matrix *mat)
{
	int i, j;
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			mat->data[i][j] = (double)rand() / (double)RAND_MAX ;
		}
	}
}

/* Fill with gaussian randoms between 0 and 1 */
void matrix_fill_gaussian_random(Matrix *mat)
{
	int i, j;
	long seed = time(NULL);
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			mat->data[i][j] = gauss0(&seed);
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
	int i, j, nc, nr, s;
	nc = b->n_cols;
	nr = a->n_rows;
	/* if (a->n_cols != b->n_rows) { */
	/* 	fprintf(stderr, "matrix_prod ERROR: cannot multiply a %dx%d matrix and a %dx%d matrix.\n", a->n_rows, a->n_cols, b->n_rows, b->n_cols); */
	/* 	return NULL; */
	/* } */
	Matrix *res = create_matrix(nr, nc);
	double val;
	for (i = 0; i < nr; i++) {
		for (j = 0; j < nc; j++) {
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
	SAME_SHAPE_CHECK("entrywise_product", "entrywise product", a, b, NULL);
	int i, j;
	Matrix *res = create_matrix(a->n_cols, a->n_rows);
	for (i = 0; i < a->n_rows; i++) {
		for (j = 0; j < a->n_cols; j++) {
			res->data[i][j] = a->data[i][j] * b->data[i][j];
		}
	}
	return res;
}

int matrix_entrywise_product(Matrix *a, Matrix *b)
{
	SAME_SHAPE_CHECK("matrix_entrywise_product", "entrywise product", a, b, 0);
	int i, j;
	for (i = 0; i < a->n_rows; i++) {
		for (j = 0; j < a->n_cols; j++) {
			a->data[i][j] *= b->data[i][j];
		}
	}
	return 1;
}

/* Add two matrices: a is altered */
int matrix_add(Matrix *a, Matrix *b)
{
	/* SAME_SHAPE_CHECK("matrix_add", "matrix addition", a, b, 0); */
	int i, j;
	for (i = 0; i < a->n_rows; i++) {
		for (j = 0; j < a->n_cols; j++) {
			a->data[i][j] += b->data[i][j];
		}
	}
}

/* Substract b from a: a is altered */
int matrix_substract(Matrix *a, Matrix *b)
{
	SAME_SHAPE_CHECK("matrix_substract", "matrix substraction", a, b,
					 0);
	int i, j;
	for (i = 0; i < a->n_rows; i++) {
		for (j = 0; j < a->n_cols; j++) {
			a->data[i][j] -= b->data[i][j];
		}
	}
}

/* Scalar product. */
void matrix_multiply(Matrix *mat, double val)
{
	int i, j;
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			mat->data[i][j] *= val;
		}
	}
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
	fprintf(stderr, "_______\n");
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			fprintf(stderr, "%f ", mat->data[i][j]);
						   /* j == (mat->n_cols - 1)? '\n': ' '); */
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "------\n");
}

void matrix_print_shape(Matrix *mat)
{
	fprintf(stderr, "[%d x %d]\n", mat->n_rows, mat->n_cols);
}

/* Turn an array of n doubles into a nx1 matrix. */
Matrix *array_to_matrix(double *array, int n)
{
	int i;
	Matrix *m = create_matrix(n, 1);
	for (i = 0; i < n; i++) {
		m->data[i][0] = array[i];
	}
	return m;
}

/* Turn a nx1 matrix into an array of n elements. */
void matrix_to_array(Matrix *mat, double *array)
{
	int i;
	for (i = 0; i < mat->n_rows; i++) {
		array[i] = mat->data[i][0];
	}
}

Matrix *transpose(Matrix *mat)
{
	int i, j;
	int rows = mat->n_rows;
	int cols = mat->n_cols;
	Matrix *T = create_matrix(cols, rows);
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			T->data[j][i] = mat->data[i][j];
		}
	}
	return T;
}

Matrix *matrix_copy(Matrix *mat)
{
	Matrix *new = COPY_MATRIX_SHAPE(mat);
	int i, j;
	for (i = 0; i < new->n_rows; i++) {
		for (j = 0; j < new->n_cols; j++) {
			new->data[i][j] = mat->data[i][j];
		}
	}
	return new;
}



/********** End matrix operations **********/
