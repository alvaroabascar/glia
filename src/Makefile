objs = lib/utils.o lib/matrix.o lib/random.o neuron.o 
progs = mnist_test tiny
CC = gcc
CFLAGS = -I. -I./lib -g -pg
LDLIBS = -lm

all:	$(progs)

clean:
	rm $(objs) $(progs)

mnist_test: $(objs)

tiny_test: $(objs)
