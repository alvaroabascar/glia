objs = neuron.o
progs = mnist_loader
CC = gcc
CFLAGS = -I./mad/matrix/ -I. -I./lib
LDLIBS = -lm

all:	$(progs)

clean:
	rm $(objs) $(progs)

mnist_loader: $(objs)
