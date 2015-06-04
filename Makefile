CPPFLAGS = -DGF_BITS=16  # -DTEST
CFLAGS = -Wall -O2 -g -mmmx -msse -msse2 -msse3 -mssse3
LDFLAGS = -g


all: fec

fec: fec.o test.o
fec.o: fec.c fec.h
test.o: test.c fec.h

.PHONY: clean
clean:
	$(RM) core *.o fec
