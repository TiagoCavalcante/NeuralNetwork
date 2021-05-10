__device__ __forceinline__ float sigmoid(const float x) {
	return 1 / (1 + expf(-x));
}