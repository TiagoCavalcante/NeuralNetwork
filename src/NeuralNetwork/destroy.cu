void destroy(
	const NeuralNetwork neuralNetwork
) {
	cudaFreeHost((float*)neuralNetwork.host_input);
	cudaFreeHost((float*)neuralNetwork.host_output);

	cudaFree((float*)neuralNetwork.device_input);
	cudaFree((float*)neuralNetwork.device_output);
	cudaFree((float*)neuralNetwork.device_weightsInput);
	cudaFree((float*)neuralNetwork.device_weightsHidden);
	cudaFree((float*)neuralNetwork.device_activatedHiddenLayer);
	cudaFree((float*)neuralNetwork.device_derivativeHiddenLayer);
	cudaFree((float*)neuralNetwork.device_activatedOutputLayer);
	cudaFree((float*)neuralNetwork.device_derivativeOutputLayer);
	cudaFree((float*)neuralNetwork.device_hiddenBias);
	cudaFree((float*)neuralNetwork.device_outputBias);

	cudaStreamDestroy(neuralNetwork.stream);

	cudaDeviceReset();
}