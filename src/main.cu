#include <stdio.h>

#include "functions/fillWeightsAndBiases.c"
#include "functions/sigmoid.cu"
#include "functions/sigmoidDerivative.cu"

#include "kernels/activateAndDerivateMatrixByVectorMultiplication.cu"
#include "kernels/activateMatrixByVectorMultiplication.cu"
#include "kernels/updateWeightsHidden.cu"
#include "kernels/updateWeightsInput.cu"

#include "NeuralNetwork/NeuralNetwork.h"
#include "NeuralNetwork/init.cu"
#include "NeuralNetwork/train.cu"
#include "NeuralNetwork/predict.cu"
#include "NeuralNetwork/destroy.cu"

int main() {
	srand(time(0));

	NeuralNetwork neuralNetwork = init(2, 2, 1, .1, 0, 16);

	for (unsigned int i = 0; i < 10000; ++i) {
		neuralNetwork.host_input[0] = .01;
		neuralNetwork.host_input[1] = .01;
		neuralNetwork.host_output[0] = .01;
		train(neuralNetwork);

		neuralNetwork.host_input[0] = .95;
		neuralNetwork.host_input[1] = .01;
		neuralNetwork.host_output[0] = .95;
		train(neuralNetwork);

		neuralNetwork.host_input[0] = .01;
		neuralNetwork.host_input[1] = .95;
		neuralNetwork.host_output[0] = .95;
		train(neuralNetwork);

		neuralNetwork.host_input[0] = .95;
		neuralNetwork.host_input[1] = .95;
		neuralNetwork.host_output[0] = .01;
		train(neuralNetwork);
	}

	// float weightsInput[4] = {200, 200, -200, -200};
	// float weightsHidden[2] = {200, 200};
	// float hiddenBias[2] = {-100, 300};
	// float outputBias[1] = {-300};

	// cudaMemcpy(
	// 	(float*)(neuralNetwork.device_weightsInput),
	// 	weightsInput,
	// 	4 * sizeof(float),
	// 	cudaMemcpyHostToDevice
	// );

	// cudaMemcpy(
	// 	(float*)(neuralNetwork.device_weightsHidden),
	// 	weightsHidden,
	// 	2 * sizeof(float),
	// 	cudaMemcpyHostToDevice
	// );

	// cudaMemcpy(
	// 	(float*)(neuralNetwork.device_hiddenBias),
	// 	hiddenBias,
	// 	2 * sizeof(float),
	// 	cudaMemcpyHostToDevice
	// );

	// cudaMemcpy(
	// 	(float*)(neuralNetwork.device_outputBias),
	// 	outputBias,
	// 	1 * sizeof(float),
	// 	cudaMemcpyHostToDevice
	// );

	neuralNetwork.host_input[0] = .01;
	neuralNetwork.host_input[1] = .01;
	predict(neuralNetwork);
	printf("X(0,0) = %f\n", neuralNetwork.host_output[0]);

	neuralNetwork.host_input[0] = .01;
	neuralNetwork.host_input[1] = .95;
	predict(neuralNetwork);
	printf("X(0,1) = %f\n", neuralNetwork.host_output[0]);

	neuralNetwork.host_input[0] = .95;
	neuralNetwork.host_input[1] = .01;
	predict(neuralNetwork);
	printf("X(1,0) = %f\n", neuralNetwork.host_output[0]);

	neuralNetwork.host_input[0] = .95;
	neuralNetwork.host_input[1] = .95;
	predict(neuralNetwork);
	printf("X(1,1) = %f\n", neuralNetwork.host_output[0]);

	float* weightsInput = (float*)malloc(neuralNetwork.hiddenNodes * neuralNetwork.inputNodes * sizeof(float));
	float* weightsHidden = (float*)malloc(neuralNetwork.outputNodes * neuralNetwork.hiddenNodes * sizeof(float));

	cudaMemcpy(
		(float*)weightsInput,
		(float*)neuralNetwork.device_weightsInput,
		neuralNetwork.hiddenNodes * neuralNetwork.inputNodes * sizeof(float),
		cudaMemcpyDeviceToHost
	);

	cudaMemcpy(
		(float*)weightsHidden,
		(float*)neuralNetwork.device_weightsHidden,
		neuralNetwork.outputNodes * neuralNetwork.hiddenNodes * sizeof(float),
		cudaMemcpyDeviceToHost
	);
	cudaDeviceSynchronize();

	printf("Wi = ");

	for (unsigned char i = 0; i < neuralNetwork.hiddenNodes * neuralNetwork.inputNodes; ++i) {
		printf("%f, ", weightsInput[i]);
	}

	printf("\nWh = ");

	for (unsigned char i = 0; i < neuralNetwork.outputNodes * neuralNetwork.hiddenNodes; ++i) {
		printf("%f, ", weightsHidden[i]);
	}

	destroy(neuralNetwork);

	return 0;
}