CC=gcc
CFLAGS=-O2 -std=c11 -Wall -Wextra
LDFLAGS=-lm

all: test_moe

test_moe: test_moe.c moe.c moe.h
	$(CC) $(CFLAGS) -o test_moe test_moe.c moe.c $(LDFLAGS)

run: test_moe
	./test_moe

clean:
	rm -f test_moe