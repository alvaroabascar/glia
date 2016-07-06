objs = ../lib/utils.o ../lib/matrix.o ../neuron.o ../lib/random.o ../lib/test_utils.o
progs = test mnist_test tiny_test
CC = gcc
CFLAGS = -I.. -I../lib -pg
LDLIBS = -lm

all:	$(progs)

clean:
	rm $(objs) $(progs)

test: $(objs)

mnist_test: $(objs)

tiny_test: $(objs)
