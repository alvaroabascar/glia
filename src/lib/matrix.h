#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix {
	int n_rows;
	int n_cols;
	double **data;
} Matrix;

typedef Matrix** MatrixList;

void matrix_fill(Matrix *mat, double value);

void matrix_fill_random(Matrix *mat);

void matrix_fill_gaussian_random(Matrix *mat);

Matrix *create_matrix(int n_rows, int n_cols);

Matrix *create_matrix_zeros(int n_rows, int n_cols);

void free_matrix(Matrix *mat);

Matrix *matrix_prod(Matrix *a, Matrix *b);

Matrix *matrix_prod_optim(Matrix *a, Matrix *b);

void matrix_multiply(Matrix *mat, double val);

Matrix *entrywise_product(Matrix *a, Matrix *b);

int matrix_entrywise_product(Matrix *a, Matrix *b);

int matrix_add(Matrix *a, Matrix *b);

int matrix_substract(Matrix *a, Matrix *b);

int matrix_cmp(Matrix *a, Matrix *b);

void matrix_assign(Matrix *mat, ...);

void matrix_print(Matrix *mat);

void matrix_print_shape(Matrix *mat);

Matrix *array_to_matrix(double *array, int n);

void matrix_to_array(Matrix *mat, double *array);

Matrix *transpose(Matrix *mat);

Matrix *matrix_copy(Matrix *mat);

#endif // MATRIX_H
