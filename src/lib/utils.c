/* Given two int arrays, copy n items from src to dst */
void arrncpy(int *dst, int *src, int n)
{
	while (--n >= 0)
		dst[n] = src[n];
}
