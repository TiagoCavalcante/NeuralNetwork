__global__ void updateWeightsHidden(
	float* __restrict__ weightsHidden,
	const float learningRate,
	const float* __restrict__ activatedHiddenLayer,
	const float* __restrict__ derivativeHiddenLayer,
	const float* __restrict__ activatedOutputLayer,
	const float* __restrict__ targetOutput,
	const unsigned short matrixRows,
	const unsigned short matrixColumns
) {
	const unsigned short row    = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned short column = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < matrixRows && column < matrixColumns) {
		weightsHidden[row * matrixColumns + column] -= learningRate * activatedHiddenLayer[column] * derivativeHiddenLayer[row] * (activatedOutputLayer[row] - targetOutput[row]);
	}
}