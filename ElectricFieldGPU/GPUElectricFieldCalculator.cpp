#include "GPUElectricFieldCalculator.h"
#include "ElectricFieldCalculatorKernel.h"
#include "cuda_runtime.h"
#include <cmath>


GPUElectricFieldCalculator::GPUElectricFieldCalculator(ChargesManager& chargesManager, int blockSize, int baseElectricForceMultiplier, int maxChargesInOneThreadRun)
	:chargesManager(chargesManager), blockSize(blockSize), baseElectricForceMultiplier(baseElectricForceMultiplier), maxChargesInOneThreadRun(maxChargesInOneThreadRun)
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
	if (xCoordinates_device != nullptr)
		cudaFree(xCoordinates_device);
	xCoordinates_device = nullptr;

	cudaError_t cudaStatus = cudaMalloc((void**)(&xCoordinates_device), chargesManager.GetXCoordinatesSize());
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	if (yCoordinates_device != nullptr)
		cudaFree(yCoordinates_device);
	yCoordinates_device = nullptr;

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
	cudaError_t cudaStatus = cudaMemset(electricFieldValues_device, 0, width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		return false;
	}

	int numberOfBlocks = static_cast<int>((width * height + blockSize - 1) / blockSize) 
		* ceil(chargesManager.GetNumberOfCharges() / static_cast<double>(maxChargesInOneThreadRun));

	LaunchStartFieldCalculation(electricFieldValues_device, width * height, xCoordinates_device, yCoordinates_device,
		chargesManager.GetNumberOfCharges(), maxChargesInOneThreadRun, baseElectricForceMultiplier, width, numberOfBlocks, blockSize);

	cudaStatus = cudaGetLastError();
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
