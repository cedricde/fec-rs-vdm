# Possible macros:
#  - -DCODE=SSE_INTRINSICS: Enable SIMD code written with SSE2 & SSSE3 intrinsics.
#  - -DCODE=VECTOR_EXTENSIONS: Enable SIMD code written with vector extensions.
#  - -DSELFTEST: Enable self test code to check various operations (very slow).
#  - -DTEST: Enable original test code.
# Notes:
#  - The SIMD instructions shall also be enabled with compiler flags.
#  - Enabling AVX on GCC makes some SSE intrinsics use AVX instructions.
#  - SIMD code is not available with GF_BITS <= 8.
CODE ?= ORIGINAL_ONLY
THREADSAFE ?= 1
OPENMP ?= 0

CPPFLAGS = -DGF_BITS=16 -DCODE=$(CODE) $(EXTRA_CPPFLAGS)
CFLAGS = -Wall -O2 -g $(EXTRA_CFLAGS)
LDFLAGS = -g $(EXTRA_LDFLAGS)

# assume x86 CPU
CFLAGS += -msse -msse2 -msse3 -mssse3


ifeq ($(THREADSAFE), 1)
	CPPFLAGS += -DTHREADSAFE
	CFLAGS += -pthread
	LDFLAGS += -pthread
endif
ifeq ($(OPENMP), 1)
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif


all: fec

fec: fec.o test.o
fec.o: fec.c fec.h
test.o: test.c fec.h

.PHONY: clean test
clean:
	$(RM) core *.o fec

test:
	@./fec
