#include <stdlib.h>
#include "matrix.h"

void fill_matrix(Matrix *mat, double value)
{
	int i, j;
	for (i = 0; i < mat->n_rows; i++) {
		for (j = 0; j < mat->n_cols; j++) {
			mat->data[i][j] = value;
		}
	}
}

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

void free_matrix(Matrix *mat)
{
	int i;
	for (i = 0; i < mat->n_rows; i++) {
		free(mat->data[i]);
	}
	free(mat->data);
	free(mat);
}
