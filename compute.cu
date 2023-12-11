#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "cuda_runtime.h"


__global__ void computePairwiseAccels(vector3 *accels, vector3 *hPos, double *mass) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NUMENTITIES && j < NUMENTITIES) {
		if (i == j) {
			FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
		} 
		else {
			vector3 distance;
			for (int k = 0; k < 3; k++) {
				distance[k] = hPos[i][k] - hPos[j][k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
			FILL_VECTOR(accels[i * NUMENTITIES + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	} 
}

__global__ void sumColsAndComputeVelAndPos(vector3* accels, vector3* hPos, vector3* hVel, vector3* accel_sum) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NUMENTITIES) {
		FILL_VECTOR(accel_sum[i], 0, 0, 0);
		for (int j = 0; j < NUMENTITIES; j++){
			for (int k = 0; k < 3; k++) {
				accel_sum[i][k] += accels[i * NUMENTITIES + j][k];
			}
		}

		for (int k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[i][k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
}


void compute() {
	vector3 *dPos, *dVel, *dAccels, *dSum;
	double *dMass;

	cudaMalloc((void**)&dPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**)&dVel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**)&dMass, sizeof(double) * NUMENTITIES);
	cudaMalloc((void**)&dAccels, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**)&dSum, sizeof(vector3) * NUMENTITIES);

	cudaMemcpy(dPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dMass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((NUMENTITIES + blockDim.x - 1) / blockDim.x, (NUMENTITIES + blockDim.y - 1) / blockDim.y);

	computePairwiseAccels<<<gridDim, blockDim>>>(dAccels, dPos, dMass);
	cudaDeviceSynchronize();

	sumColsAndComputeVelAndPos<<<gridDim, blockDim>>>(dAccels, dPos, dVel, dSum);
	cudaDeviceSynchronize();

	cudaMemcpy(hPos, dPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(dPos);
	cudaFree(dVel);
	cudaFree(dMass);
	cudaFree(dAccels);
}
