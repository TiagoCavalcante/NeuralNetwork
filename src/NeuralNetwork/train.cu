void train(
	const NeuralNetwork neuralNetwork
) {
	cudaMemcpy(
		(float*)neuralNetwork.device_input,
		neuralNetwork.host_input,
		neuralNetwork.inputNodes * sizeof(float),
		cudaMemcpyHostToDevice
	);

	cudaMemcpyAsync(
		(float*)neuralNetwork.device_output,
		neuralNetwork.host_output,
		neuralNetwork.outputNodes * sizeof(float),
		cudaMemcpyHostToDevice,
		neuralNetwork.stream
	);

	activateAndDeriveMatrixByVectorMultiplication <<<
		neuralNetwork.hiddenLayerBlocksPerGrid,
		neuralNetwork.threadsPerBlock
	>>> (
		(float*)neuralNetwork.device_activatedHiddenLayer,
		(float*)neuralNetwork.device_derivativeHiddenLayer,
		neuralNetwork.device_input,
		neuralNetwork.device_weightsInput,
		neuralNetwork.device_hiddenBias,
		neuralNetwork.hiddenNodes,
		neuralNetwork.inputNodes
	);

	activateAndDeriveMatrixByVectorMultiplication <<<
		neuralNetwork.outputBlocksPerGrid,
		neuralNetwork.threadsPerBlock
	>>> (
		(float*)neuralNetwork.device_activatedOutputLayer,
		(float*)neuralNetwork.device_derivativeOutputLayer,
		neuralNetwork.device_activatedHiddenLayer,
		neuralNetwork.device_weightsHidden,
		neuralNetwork.device_outputBias,
		neuralNetwork.outputNodes,
		neuralNetwork.hiddenNodes
	);

	// next kernel operates with the results of the last one
	cudaDeviceSynchronize();

	updateWeightsHidden <<<
		neuralNetwork.weightsHiddenBlocksPerGrid,
		neuralNetwork.threadsPerBlock2d,
		0,
		neuralNetwork.stream
	>>> (
		(float*)neuralNetwork.device_weightsHidden,
		neuralNetwork.learningRate,
		neuralNetwork.device_activatedHiddenLayer,
		neuralNetwork.device_derivativeHiddenLayer,
		neuralNetwork.device_activatedOutputLayer,
		neuralNetwork.device_output,
		neuralNetwork.outputNodes,
		neuralNetwork.hiddenNodes
	);

	updateWeightsInput <<<
		neuralNetwork.weightsInputBlocksPerGrid,
		neuralNetwork.threadsPerBlock2d,
		0,
		neuralNetwork.stream
	>>> (
		(float*)neuralNetwork.device_weightsInput,
		neuralNetwork.learningRate,
		neuralNetwork.device_input,
		neuralNetwork.device_derivativeHiddenLayer,
		neuralNetwork.device_activatedOutputLayer,
		neuralNetwork.device_derivativeOutputLayer,
		neuralNetwork.device_output,
		neuralNetwork.inputNodes,
		neuralNetwork.hiddenNodes,
		neuralNetwork.outputNodes
	);

	// next epoch need that operations have been finished
	cudaDeviceSynchronize();
}