#include <stdio.h>

#define TEST_RES 0
#define ASSERT(MSG, ...) \
	if ( __VA_ARGS__ ) { \
		printf("[OK]    %s\n", MSG); \
	} else { \
		printf("[FAIL]  %s\n", MSG); \
	}
