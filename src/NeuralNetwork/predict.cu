void predict(
	const NeuralNetwork neuralNetwork
) {
	cudaMemcpy(
		(float*)neuralNetwork.device_input,
		(float*)neuralNetwork.host_input,
		neuralNetwork.inputNodes * sizeof(float),
		cudaMemcpyHostToDevice
	);

	activateMatrixByVectorMultiplication <<<
		neuralNetwork.hiddenLayerBlocksPerGrid,
		neuralNetwork.threadsPerBlock
	>>> (
		(float*)neuralNetwork.device_activatedHiddenLayer,
		neuralNetwork.device_input,
		neuralNetwork.device_weightsInput,
		neuralNetwork.device_hiddenBias,
		neuralNetwork.hiddenNodes,
		neuralNetwork.inputNodes
	);

	activateMatrixByVectorMultiplication <<<
		neuralNetwork.outputBlocksPerGrid,
		neuralNetwork.threadsPerBlock
	>>> (
		(float*)neuralNetwork.device_output,
		(float*)neuralNetwork.device_activatedHiddenLayer,
		neuralNetwork.device_weightsHidden,
		neuralNetwork.device_outputBias,
		neuralNetwork.outputNodes,
		neuralNetwork.hiddenNodes
	);

	cudaMemcpy(
		(float*)neuralNetwork.host_output,
		neuralNetwork.device_output,
		neuralNetwork.outputNodes * sizeof(float),
		cudaMemcpyDeviceToHost
	);

	cudaDeviceSynchronize();
}