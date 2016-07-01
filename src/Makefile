objs = lib/utils.o lib/matrix.o lib/random.o neuron.o 
progs = mnist_loader
CC = gcc
CFLAGS = -I. -I./lib -g -pg
LDLIBS = -lm

all:	$(progs)

clean:
	rm $(objs) $(progs)

mnist_loader: $(objs)
