# CUDARMAT

CUDARMAT generates an R-MAT test graph using CUDA on nVidia GPUs and
outputs the graph in one of the following formats:

- host-endian binary edge list,
- textual DIMACS format, and
- host-endian STINGER graph and dynamic edge action list.

The Makefile depends on CUDA SDK makefiles. Pass --help to the built
executable for more argument details.
