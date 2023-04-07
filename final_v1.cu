#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;

// #define show_output

void printGPUMemmory() {
	int device;	
	size_t free, total;
	cudaGetDevice(&device);
	cudaMemGetInfo ( &free, &total );
	printf("GPU %u Memory left:\t%llu / %llu\n", device, free, total);
}

void read_input(const char* filename, bool mode, int& U, int& M, int& N, 
	multimap<int, pair<int, double>>& m_u_ratings) { 
	std::ifstream fin(filename);

	if (mode == 0) fin >> U >> M >> N;
	else fin >> M >> U >> N;

	for (int i = 0; i < N; i++) {
		int u = 0, m = 0;
		double r = 0;
		if (mode == 0) fin >> u >> m >> r;
		else fin >> m >> u >> r;
		pair<int, double> u_r (u, r);
		m_u_ratings.insert({m, u_r});
	}
}

void split(int* movieList, int* userCount, int* userId, double* u_ratings, 
	multimap<int, pair<int, double>>& m_u_ratings, int N) {
	int i = 0, m = 0;
	movieList[0] = 0;	
	for (auto it = m_u_ratings.begin(); it != m_u_ratings.end(); it++, i++) {
		if (i == 0) movieList[m] = it->first;
		if (it->first != movieList[m]) {
			userCount[m] = i;
			m++;
			movieList[m] = it->first;
		}
		userCount[m] = N;
		userId[i] = it->second.first;
		u_ratings[i] = it->second.second;
	}
	m_u_ratings.clear();
}

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__
void calculateMagnitude(double* u_ratings, int* userCount, double* magnitudes, int M) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  	unsigned int stride = blockDim.x * gridDim.x;

	for(unsigned int m = index; m < M; m += stride) {
		/* Calculate Mean */
		int start = (m > 0) ? userCount[m-1] : 0;
		double mean = 0;
		for (int i = start; i < userCount[m]; i++) {
			mean += u_ratings[i];
		}
		mean /= (double) userCount[m] - start;

		/* Calculate Magnitude */
		magnitudes[m] = 0;
		for (int i = start; i < userCount[m]; i++) {
			u_ratings[i] -= mean;
			magnitudes[m] += u_ratings[i] * u_ratings[i];
		}
		magnitudes[m] = sqrt(magnitudes[m]);
	}
}

__global__
void calculateSimilarity(int* userId, double* u_ratings, int* userCount, double* magnitudes, double* similarities, int M) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int unit = M / stride + 1;
	
	for(unsigned int m = index; m < M; m += stride) {
		for(int n = m + 1; n < M; n++) {
			if (((n + m) % 2 && n < m) || ((n + m) % 2 == 0 && n > m)) {
				double mag_prod = magnitudes[m] * magnitudes[n];
				if (mag_prod == 0) {
					similarities[m*M + n] = 0;
					similarities[n*M + m] = 0;
				}
				else {
					int i = (m > 0) ? userCount[m-1] : 0;
					int j = userCount[n-1];
					double dot_prod = 0;

					while (i < userCount[m] && j < userCount[n]) {
						while (userId[i] < userId[j]) i++;
						while (userId[i] > userId[j]) j++;
						if (userId[i] == userId[j]) {
							dot_prod += u_ratings[i] * u_ratings[j];
							i++, j++;
						}
					}
					similarities[m*M + n] = dot_prod / mag_prod;
					similarities[n*M + m] = dot_prod / mag_prod;
				}
			}
		}
	}
}

__global__ 
void predictRatings(double *predictions, double *similarities, int *movieCount, int *movieList, int *movieId, int *m_ratings, int U, int M) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int unit = U / stride + 1;

	/* Sync movieId to match index of similarities matrix */
	for(unsigned int u = index + 1; u < U + 1; u += stride) {
		int start = (u > 0)? movieCount[u-1] : 0;
		int i = start, m = 0;

		while (i < movieCount[u] && m < M) {
			if(movieId[i] == movieList[m]) {
				movieId[i] = m;
				i++, m++;
			}
			while(movieId[i] < movieList[m]) i++;
			while(movieId[i] > movieList[m]) m++;
		}

		/* Predict rating */
		m = 1;
		for(i = start; i < movieCount[u]; i++) {
			for (; m < M && m < movieId[i]; m++) {
				double sum_sim = 0, sum_rating = 0;
				for (int n = start; n < movieCount[u]; n++) {
					if (similarities[m*M + movieId[n]] > 0) {
						sum_sim += similarities[m*M + movieId[n]];
						sum_rating += similarities[m*M + movieId[n]] * m_ratings[n];
					}
				}
				double pred = (sum_sim == 0)? 0 : sum_rating / sum_sim;
				// predictions[u*U + movieId[m]] = (sum_sim == 0)? 0 : sum_rating / sum_sim;
				if (predict) printf("%d %d %f\n", u, movieList[m], pred);
			}
			// if (m < M) predictions[u*U + movieId[m]] = -1;
			m++;
		}
	}
}

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "Please provide input file\n");
		return 1;
	}
	double totalTime = 0;
	int U, M, N;
	multimap<int, pair<int, double>> m_u_ratings;
	read_input(argv[1], argv[2], U, M, N, m_u_ratings);

	/* Split */
	int *movieList;
	int *userCount;
	int *userId;
	double *u_ratings;    
	double *magnitudes;
    checkCuda( cudaMallocManaged(&movieList, sizeof(int) * M) );
    checkCuda( cudaMallocManaged(&userCount, sizeof(int) * M) );
    checkCuda( cudaMallocManaged(&magnitudes, sizeof(double) * M) );
    checkCuda( cudaMallocManaged(&userId, sizeof(int) * N) );
    checkCuda( cudaMallocManaged(&u_ratings, sizeof(double) * N) );
	
    split(movieList, userCount, userId, u_ratings, m_u_ratings, N);

	/* Calculate Magnitude */
    clock_t start = clock();
	calculateMagnitude<<<1, 1024>>>(u_ratings, userCount, magnitudes, M);
	checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
	clock_t end = clock();
    double elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
	totalTime += elapsed;
	if (eval) {
		printf("Calculate Magnitude [1] (ms): %f\n", elapsed);
		printGPUMemmory();
	}
	
	/* Calculate Similarities */
	double* similarities;
	checkCuda( cudaMallocManaged(&similarities, sizeof(double) * M* (M + M)) );
	
	start = clock();
	calculateSimilarity<<<16, 128>>>(userId, u_ratings, userCount, magnitudes, similarities, M);	
	checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
	end = clock();
    totalTime += elapsed;
	if (eval) {
		printf("Calculate Similarities [1] (ms): %f\n", elapsed);
		printGPUMemmory();
	}

	// checkCuda( cudaFree(userCount) );
	// checkCuda( cudaFree(magnitudes) );
	// checkCuda( cudaFree(userId) );
	// checkCuda( cudaFree(u_ratings) );
	// checkCuda( cudaFree(similarities) );
	
	/* Predict Ratings */
	int* m_ratings;
	int* movieId;
	int* movieCount;
	double* predictions;
	checkCuda( cudaMallocManaged(&movieCount, sizeof(int) * U) );
    checkCuda( cudaMallocManaged(&movieId, sizeof(int) * N) );
    checkCuda( cudaMallocManaged(&m_ratings, sizeof(int) * N) );
    checkCuda( cudaMallocManaged(&predictions, sizeof(double) * N * M) );
	
	std::ifstream fin(argv[1]);
	movieCount[0] = 0;
	int temp;
	fin >> temp >> temp >> temp;
	for (int i = 0; i < N; i++) {
		int u = 0, m = 0;
		double r = 0;
		if (argv[2] == 0) fin >> u >> m >> r;
		else fin >> u >> m >> r;
		movieId[i] = m;
		m_ratings[i] = r;
		movieCount[u] = i + 1;
	}

	start = clock();
	predictRatings<<<32, 64>>>(predictions, similarities, movieCount, movieList, movieId, m_ratings, U, M);
	checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
	end = clock();
    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;totalTime += elapsed;
    if (eval) {
		printGPUMemmory();
		printf("Predict Ratings [1] (ms): %f\n", elapsed);
    	printf("Total time (ms): %f\n", totalTime);
	}
}

