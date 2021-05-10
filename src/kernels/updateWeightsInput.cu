__global__ void updateWeightsInput(
	float* __restrict__ weightsInput,
	const float learningRate,
	const float* __restrict__ input,
	const float* __restrict__ derivativeHiddenLayer,
	const float* __restrict__ activatedOutputLayer,
	const float* __restrict__ derivativeOutputLayer,
	const float* __restrict__ targetOutput,
	const unsigned short inputNodes,
	const unsigned short hiddenNodes,
	const unsigned short outputNodes
) {
	const unsigned short row    = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned short column = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < hiddenNodes && column < inputNodes) {
		float sum1 = 0, sum2 = 0;

		for (unsigned short i = 0; i < outputNodes; ++i) {
			sum1 += derivativeOutputLayer[i] * weightsInput[row * inputNodes + i] * (activatedOutputLayer[i] - targetOutput[i]);
		}

		for (unsigned short i = 0; i < inputNodes; ++i) {
			sum2 += input[i] * weightsInput[i * inputNodes + column];
		}

		// weightsInput[row * inputNodes + column] += learningRate * input[column] * derivativeHiddenLayer[row] * sum;
		weightsInput[row * inputNodes + column] -= learningRate * input[column] * sum1 * (sigmoid(sum2) * (1 - sigmoid(sum2)));
	}
}