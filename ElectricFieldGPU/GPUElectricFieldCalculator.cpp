#include "GPUElectricFieldCalculator.h"
#include "ElectricFieldCalculatorKernel.h"
#include "cuda_runtime.h"


GPUElectricFieldCalculator::GPUElectricFieldCalculator(const ChargesManager& chargesManager, int blockSize, int baseElectricForceMultiplier)
	:chargesManager(chargesManager), blockSize(blockSize), baseElectricForceMultiplier(baseElectricForceMultiplier)
{
	electricFieldValues_device = 0;
	xCoordinates_device = 0;
	yCoordinates_device = 0;
}

bool GPUElectricFieldCalculator::SetDevice(int id)
{
	cudaError_t cudaStatus = cudaSetDevice(id);
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	return true;
}

bool GPUElectricFieldCalculator::CreateDeviceElectricFieldMatrix(int width, int height)
{
	this->width = width;
	this->height = height;

	if (electricFieldValues_device != nullptr)
		cudaFree(electricFieldValues_device);
	electricFieldValues_device = nullptr;

	cudaError_t cudaStatus = cudaMalloc((void**)&electricFieldValues_device, width * height * sizeof(float));
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
		chargesManager.GetNumberOfCharges(), baseElectricForceMultiplier, width, numberOfBlocks, blockSize);

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

	cudaStatus = cudaMemcpy(electricFieldMatrix, electricFieldValues_device, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	return true;
}

GPUElectricFieldCalculator::~GPUElectricFieldCalculator()
{
	cudaDeviceReset();

	cudaFree(electricFieldValues_device);
	cudaFree(xCoordinates_device);
	cudaFree(yCoordinates_device);
}
