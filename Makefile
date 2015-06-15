# Possible flags:
#  - -DENABLE_SSE_INTRIN: Enable SIMD code written with SSE2 & SSSE3 intrinsics.
#  - -DENABLE_VECTOR_EXT: Enable SIMD code written with vector extensions.
#  - -DSELFTEST: Enable self test code to check various operations (very slow).
# Notes:
#  - Only one SIMD code can be enabled at a time.
#  - The SIMD instructions shall also be enabled with compiler flags.
#  - Enabling AVX on GCC makes some SSE intrinsics use AVX instructions.
#  - SIMD code is not available with GF_BITS <= 8.
CPPFLAGS = -DGF_BITS=16 -DENABLE_SSE_INTRIN  # -DTEST
CFLAGS = -Wall -O2 -g -msse -msse2 -msse3 -mssse3
LDFLAGS = -g


all: fec

fec: fec.o test.o
fec.o: fec.c fec.h
test.o: test.c fec.h

.PHONY: clean
clean:
	$(RM) core *.o fec
