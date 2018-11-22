#include "GPUElectricFieldCalculator.h"
#include "ElectricFieldCalculatorKernel.h"


GPUElectricFieldCalculator::GPUElectricFieldCalculator(ChargesManager chargesManager, int blockSize)
	:chargesManager(chargesManager), blockSize(blockSize)
{
	electricFieldValues_device = 0;
	xCoordinates_device = 0;
	yCoordinates_device = 0;
}

bool GPUElectricFieldCalculator::SetDevice(int id)
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	return true;
}

bool GPUElectricFieldCalculator::CreateDeviceElectricFieldMatrix(int width, int height)
{
	this->width = width;
	this->height = height;

	cudaError_t cudaStatus = cudaMalloc((void**)&electricFieldValues_device, width * height * sizeof(*electricFieldValues_device));
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	return true;
}

bool GPUElectricFieldCalculator::CreateDeviceChargesArrays()
{
	cudaError_t cudaStatus = cudaMalloc((void**)(&xCoordinates_device), chargesManager.GetXCoordinatesSize());
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	cudaStatus = cudaMalloc((void**)(&yCoordinates_device), chargesManager.GetYCoordinatesSize());
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	return true;
}

bool GPUElectricFieldCalculator::UpdateDeviceChargesArrays()
{
	cudaError_t cudaStatus = cudaMemcpy(xCoordinates_device, chargesManager.GetXCoordinates(), chargesManager.GetXCoordinatesSize(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	cudaStatus = cudaMemcpy(yCoordinates_device, chargesManager.GetYCoordinates(), chargesManager.GetYCoordinatesSize(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	return true;
}

bool GPUElectricFieldCalculator::StartCalculatingElectricField()
{
	int numberOfBlocks = ((width * height + blockSize - 1) / blockSize);

	LaunchStartFieldCalculation(electricFieldValues_device, width * height, xCoordinates_device, yCoordinates_device,
		chargesManager.GetNumberOfCharges(), 1, width, numberOfBlocks, blockSize);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	return true;
}

bool GPUElectricFieldCalculator::SynchronizeDeviceAndCopyResult(float* electricFieldMatrix)
{
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	cudaStatus = cudaMemcpy(electricFieldMatrix, electricFieldValues_device, width * height * sizeof(*electricFieldValues_device), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		return false;
	}
}
