#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix {
	int n_rows;
	int n_cols;
	double **data;
} Matrix;

typedef Matrix** MatrixList;

void fill_matrix(Matrix *mat, double value);

Matrix *create_matrix(int n_rows, int n_cols);

void free_matrix(Matrix *mat);

Matrix *matrix_prod(Matrix *a, Matrix *b);

Matrix *entrywise_product(Matrix *a, Matrix *b);

Matrix *matrix_add(Matrix *a, Matrix *b);

int matrix_cmp(Matrix *a, Matrix *b);

void matrix_assign(Matrix *mat, ...);

void matrix_print(Matrix *mat);

Matrix *array_to_matrix(double *array, int n);

void matrix_to_array(Matrix *mat, double *array);

#endif // MATRIX_H
