__global__ void activateAndDeriveMatrixByVectorMultiplication(
	float* __restrict__ resultVector,
	float* __restrict__ resultVectorDerivative,
	const float* __restrict__ vector,
	const float* __restrict__ matrix,
	const float* __restrict__ bias,
	const unsigned short matrixRows,
	const unsigned short matrixColumns
) {
	const unsigned short elementNumber = blockIdx.x * blockDim.x + threadIdx.x;

	if(elementNumber < matrixRows) {
		float sum = 0;

		for (unsigned short i = 0; i < matrixColumns; ++i) {
			sum += vector[i] * matrix[elementNumber * matrixColumns + i];
		}

		resultVector[elementNumber] = sigmoid(sum + bias[elementNumber]);
		resultVectorDerivative[elementNumber] = sigmoidDerivative(sum);
	}
}