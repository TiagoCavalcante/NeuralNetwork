__device__ __forceinline__ float sigmoidDerivative(const float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}