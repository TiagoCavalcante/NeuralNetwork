/**
 * initialize NeuralNetwork struct and allocate all variables on GPU memory
 * @param cudaDevice - GPU number, 1st GPU is 0
 * @returns a pointer to the new NeuralNetowrk structure or 0 if it fails
*/
NeuralNetwork init(
	const unsigned short inputNodes,
	const unsigned short hiddenNodes,
	const unsigned short outputNodes,
	float learningRate,
	unsigned char cudaDevice,
	const unsigned short threadsPerBlock
) {
	if (cudaSetDevice(cudaDevice) != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");

		return {};
	}

	dim3 threadsPerBlock2d(threadsPerBlock, threadsPerBlock);

	dim3 weightsInputBlocksPerGrid(
		(unsigned short)ceil((float)hiddenNodes / (float)threadsPerBlock),
		(unsigned short)ceil((float)inputNodes  / (float)threadsPerBlock)
	);
	dim3 weightsHiddenBlocksPerGrid(
		(unsigned short)ceil((float)outputNodes / (float)threadsPerBlock),
		(unsigned short)ceil((float)hiddenNodes / (float)threadsPerBlock)
	);

	float* host_input;
	float* host_output;

	float* device_input;
	float* device_output;
	float* device_weightsInput;
	float* device_weightsHidden;
	float* device_activatedHiddenLayer;
	float* device_derivativeHiddenLayer;
	float* device_activatedOutputLayer;
	float* device_derivativeOutputLayer;
	float* device_hiddenBias;
	float* device_outputBias;

	if (cudaMallocHost(
		&host_input,
		inputNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMallocHost(
		&host_output,
		outputNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_input,
		inputNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_output,
		outputNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_weightsInput,
		hiddenNodes * inputNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_weightsHidden,
		outputNodes * hiddenNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_activatedHiddenLayer,
		hiddenNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_derivativeHiddenLayer,
		hiddenNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_activatedOutputLayer,
		outputNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_derivativeOutputLayer,
		outputNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_hiddenBias,
		hiddenNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	if (cudaMalloc(
		&device_outputBias,
		outputNodes * sizeof(float)
	) != cudaSuccess) goto MallocError;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	fillWeightsAndBiases(
		device_weightsInput,
		device_weightsHidden,
		device_hiddenBias,
		device_outputBias,
		inputNodes,
		hiddenNodes,
		outputNodes
	);

	return {
		inputNodes,
		hiddenNodes,
		outputNodes,

		learningRate,

		stream,

		threadsPerBlock,
		threadsPerBlock2d,

		// hiddenLayerBlocksPerGrid
		(unsigned short)ceil((float)hiddenNodes / (float)threadsPerBlock),
		// outputBlocksPerGrid
		(unsigned short)ceil((float)outputNodes / (float)threadsPerBlock),

		weightsInputBlocksPerGrid,
		weightsHiddenBlocksPerGrid,

		host_input,
		host_output,

		device_input,
		device_output,
		device_weightsInput,
		device_weightsHidden,
		device_activatedHiddenLayer,
		device_derivativeHiddenLayer,
		device_activatedOutputLayer,
		device_derivativeOutputLayer,
		device_hiddenBias,
		device_outputBias
	};

MallocError:
	fprintf(stderr, "Memory is leaking!");

	cudaFreeHost(host_input);
	cudaFreeHost(host_output);

	cudaFree(device_input);
	cudaFree(device_output);
	cudaFree(device_weightsInput);
	cudaFree(device_weightsHidden);
	cudaFree(device_activatedHiddenLayer);
	cudaFree(device_derivativeHiddenLayer);
	cudaFree(device_activatedOutputLayer);
	cudaFree(device_derivativeOutputLayer);
	cudaFree(device_hiddenBias);
	cudaFree(device_outputBias);

	return {};
}