objs = ../lib/utils.o ../lib/matrix.o ../neuron.o ../lib/random.o ../lib/test_utils.o
progs = test
CC = gcc
CFLAGS = -I.. -I../lib -g
LDLIBS = -lm

all:	$(progs)

clean:
	rm $(objs) $(progs)

test: $(objs)
