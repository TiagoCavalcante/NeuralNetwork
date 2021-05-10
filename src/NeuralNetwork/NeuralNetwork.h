typedef struct {
	const unsigned short inputNodes;
	const unsigned short hiddenNodes;
	const unsigned short outputNodes;

	const float learningRate;

	const cudaStream_t stream;

	const unsigned short threadsPerBlock;
	const dim3 threadsPerBlock2d;

	const unsigned short hiddenLayerBlocksPerGrid;
	const unsigned short outputBlocksPerGrid;

	const dim3 weightsInputBlocksPerGrid;
	const dim3 weightsHiddenBlocksPerGrid;

	float* host_input;
	float* host_output;

	const float* device_input;
	const float* device_output;
	const float* device_weightsInput;
	const float* device_weightsHidden;
	const float* device_activatedHiddenLayer;
	const float* device_derivativeHiddenLayer;
	const float* device_activatedOutputLayer;
	const float* device_derivativeOutputLayer;
	const float* device_hiddenBias;
	const float* device_outputBias;
} NeuralNetwork;