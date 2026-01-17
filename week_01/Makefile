CC=cc
CFLAGS=-O3 -Wall -Wextra -std=c11
LDFLAGS=-pthread

all: test benchmark

test: test.o matmul.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

benchmark: benchmark.o matmul.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c matmul.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o test benchmark
