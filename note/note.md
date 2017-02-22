

## Lesson 2 GPU Hardware and Parallel Communication Patterns

### Parallel communication patterns

1. __Map__: Task read from and write to specific data elements
2. __Gather__: Each calculation gathers input data elements together from different places to compute an output result
3. __Scatter__: Each parallel task needs to write its result in a different place or in multiple places
4. __Stencil__: Tasks read input from a fixed neighborhood in an array
5. __Transpose__: Tasks reorder data elements in memory
   1. AOS: Array of structures
   2. SOA: Structure of arrays

__Note__: Difference between __Stencil__ and __Gather__: __Stencil__ require tasks for all elements, while __Gather__ does not.

| Map        | Transpose  | Gather      | Scatter     | Stencil        | Reduce     | Scan/Sort  |
| ---------- | ---------- | ----------- | ----------- | -------------- | ---------- | ---------- |
| one-to-one | one-to-one | many-to-one | one-to-many | several-to-one | all-to-one | all-to-all |

![patterns](source/patterns.png)

### Thread Blocks and GPU Hardware

1. GPUs have __Streaming Multiprocessors__ (maybe 1, maybe 16), they run in parallel and independently
2. A __Streaming Processors__ have __simple processors__ and __memory__.
3. __GPU__ is responsible for allocating __thread blocks__ to __SM__s.
4. __Programmer__ only have to worry about giving the __GPU__ a big pile of __thread blocks__.

![gpu_sm_threadblock](source/gpu_sm_threadblock.png)

### Memory Model

1. __local memory__
2. __shared memory__
3. __global memory__
4. __host memory__

![memory](source/memory.png)

### Synchronization - Barrier

```cpp
__global__ void foo(){
	__shared__ int s[1024];
	int i = threadIdx.x;
  	__syncthreads(); // A barrier
  	int temp = s[i-1];
  	__syncthreads(); // A barrier
  	s[i] = temp;
	__syncthreads(); // A barrier
  	printf(...)
}
```

