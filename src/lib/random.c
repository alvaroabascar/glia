#include <stdlib.h> // For random(), RAND_MAX
#include <math.h>


#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

/* return uniformly distributed random numbers */
float rand0(long *seed) {
    long k;
    float ans;

    *seed ^= MASK;
    k = (*seed)/IQ;
    *seed = IA * (*seed - k*IQ) - IR*k;
    if (*seed < 0) {
        * seed += IM;
    }
    ans = AM*(*seed);
    *seed ^= MASK;
    return ans;
}

/* boxmuller transform. Turns uniform random numbers into
 * normally distributed random numbers, with mean 0 and
 * standard deviation 1.
 */
float gauss0(long *seed)
{
    float x1, x2, y1, r, fac;
    static float y2;
    static int flag = 0;

    if (flag) {
        flag = 0;
        return y2;
    }
    do {
        x1 = rand0(seed)*2 - 1;
        x2 = rand0(seed)*2 - 1;
        r = x1*x1 + x2*x2;
    } while (r >= 1);
    fac = sqrt((-2*log(r)/r));
    y1 = x1*fac;
    y2 = x2*fac;
    flag = 1;
    return y1;
}

/* return a random number between 0 and limit (inclusive). */
int rand_lim(int limit)
{

  int divisor = RAND_MAX / (limit + 1);
  int retval;

  do {
    retval = rand() / divisor;
  } while (retval > limit);

  return retval;
}

/* Random within a range of ints (both ends included) */
long random_in_range(long min, long max)
{
    return min + rand_lim(max - min);
}
