void fillWeightsAndBiases(
	const float* __restrict__ device_weightsInput,
	const float* __restrict__ device_weightsHidden,
	const float* __restrict__ device_hiddenBias,
	const float* __restrict__ device_outputBias,
	const unsigned short inputNodes,
	const unsigned short hiddenNodes,
	const unsigned short outputNodes
) {
	float* weightsInput = (float*)malloc(hiddenNodes * inputNodes * sizeof(float));
	float* weightsHidden = (float*)malloc(outputNodes * hiddenNodes * sizeof(float));
	float* hiddenBias = (float*)malloc(hiddenNodes * sizeof(float));
	float* outputBias = (float*)malloc(outputNodes * sizeof(float));

	for (unsigned int i = 0; i < hiddenNodes * inputNodes; ++i) {
		// random number between -1 and 1
		weightsInput[i] = (float)rand() / RAND_MAX * 2 - 1;
	}

	for (unsigned int i = 0; i < outputNodes * hiddenNodes; ++i) {
		// random number between -1 and 1
		weightsHidden[i] = (float)rand() / RAND_MAX * 2 - 1;
	}

	for (unsigned int i = 0; i < hiddenNodes; ++i) {
		// random number between -0.5 and 0.5
		// hiddenBias[i] = (float)rand() / RAND_MAX - .5;
		hiddenBias[i] = 0;
	}

	for (unsigned int i = 0; i < outputNodes; ++i) {
		// random number between -0.5 and 0.5
		// outputBias[i] = (float)rand() / RAND_MAX - .5;
		outputBias[i] = 0;
	}

	cudaMemcpy(
		(float*)device_weightsInput,
		weightsInput,
		hiddenNodes * inputNodes * sizeof(float),
		cudaMemcpyHostToDevice
	);

	cudaMemcpy(
		(float*)device_weightsHidden,
		weightsHidden,
		outputNodes * hiddenNodes * sizeof(float),
		cudaMemcpyHostToDevice
	);

	cudaMemcpy(
		(float*)device_hiddenBias,
		hiddenBias,
		hiddenNodes * sizeof(float),
		cudaMemcpyHostToDevice
	);

	cudaMemcpy(
		(float*)device_outputBias,
		outputBias,
		outputNodes * sizeof(float),
		cudaMemcpyHostToDevice
	);
}