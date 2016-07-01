/* Given two int arrays, copy n items from src to dst */
void arrncpy(int *dst, int *src, int n)
{
	while (--n >= 0)
		dst[n] = src[n];
}
/* Given two int arrays, copy n items from src to dst */
void arrncpy_double(double *dst, double *src, int n)
{
	while (--n >= 0)
		dst[n] = src[n];
}

/* Return index of the maximum of an array */
int argmax(double *array, int n)
{
	double max = 0;
	int index = n;
	for (; n >= 0; n--) {
		if (array[n] > max) {
			max = array[n];
			index = n;
		}
	}
	return index;
}
