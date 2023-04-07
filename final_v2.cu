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

struct Rating {
	int id;
	double value;
};

struct Profile1 {
	int count;
	double magnitude;
	Rating* ratings;
	// double* similarities;
};

struct Profile2 {
	int count;
	Rating* ratings;
};

void printGPUMemmory() {
	int device;	
	size_t free, total;
	cudaGetDevice(&device);
	cudaMemGetInfo ( &free, &total );
	printf("GPU %u Memory left:\t%llu / %llu\n", device, free, total);
}

void readInput(const char* filename, bool mode, multimap<int, Rating>& ratingsMap) { 
	std::ifstream fin(filename);

	int U, M, N;
	fin >> U >> M >> N;
	// printf("%s %d %d %d\n", filename, U, M, N);
	
	for (int i = 0; i < N; i++) {
		int k1 = 0, k2 = 0;
		double v = 0;
		if (mode == 0) fin >> k1 >> k2 >> v;
		else fin >> k2 >> k1 >> v;
		// printf("%d %d %f\n", k1, k2, v);
		// printf("%d\n",mode);
		Rating r;
		r.id = k2;
		r.value = v;
		ratingsMap.insert({k1, r});
		// printf("%d u%d r%f\n", k1, r.id, r.value);
	}
}

template <class T>
void mapMovie(const char* filename, T** mData, int* movieList) {
	// for (auto it = m_u_ratings.begin(); it != m_u_ratings.end(); it++) {
	// 	printf("%d u%d r%f\n", it->first, it->second.id, it->second.value);
	// }
	multimap<int, Rating> m_u_ratings;
	readInput(filename, 1, m_u_ratings);

	int i = 0, m = -1;	
	for (auto it = m_u_ratings.begin(); it != m_u_ratings.end(); it++, i++) {
		if (m == -1 || it->first != movieList[m]) {
			// count.push_back(i);
			// if (m > -1) printf("%d %d %d\n", m, movieList[m], it->first);
			i = 0, m++;
			int cnt = m_u_ratings.count(it->first);
			// cudaMallocManaged(&mData[m], sizeof(T) + sizeof(Rating) * cnt + sizeof(double) * M * 2);
			cudaMallocManaged(&mData[m], sizeof(T) + sizeof(Rating) * cnt);
			mData[m]->count = cnt;
			mData[m]->ratings = (Rating*) mData[m] + sizeof(mData);
			// mData[m]->similarities = (double*) mData[m]->ratings + sizeof(Rating) * cnt;
			movieList[m] = it->first;
			// printf("%d %d %d %d\n", m, movieList[m], it->first, mData[m]->count);
		}
		mData[m]->ratings[i].id = it->second.id;
		mData[m]->ratings[i].value = it->second.value;
		// printf("%d %f u%d r%f\n", mData[m]->ratings[i].id, mData[m]->ratings[i].value, it->second.id, it->second.value);
	}
	// printf("%d", m);
	// for (int m = 0; m < M; m++) {
	// 	if(mData[m]) {
	// 		printf("%d m%d %d\n", m, movieList[m], mData[m]->count);
	// 		for (int i = 0; i < mData[m]->count; i++) {
	// 			printf("u%d %f", mData[m]->ratings[i].id, mData[m]->ratings[i].value);
	// 		}
	// 		// for (int i = 0; i < M; i++) {
	// 		// 	mData[m]->similarities[i] = 0;
	// 		// 	printf("%d %d %f\n", m, i, mData[m]->similarities[i]);
	// 		// }
	// 		printf("\n");
	// 	}
	// }
	// printf("\n");
}

template <class T>
void mapUser(const char* filename, T** uData) {
	multimap<int, Rating> u_m_ratings;
	readInput(filename, 0, u_m_ratings);
	// for (auto it = u_m_ratings.begin(); it != u_m_ratings.end(); it++) {
	// 	printf("%d u%d r%f\n", it->first, it->second.id, it->second.value);
	// }

	int i = 0, u = -1;	
	for (auto it = u_m_ratings.begin(); it != u_m_ratings.end(); it++, i++) {
		if (u == -1 || it->first != u) {
			i = 0, u = it->first;
			int cnt = u_m_ratings.count(u);
			checkCuda( cudaMallocManaged(&uData[u], sizeof(T) + sizeof(Rating) * cnt));
			uData[u]->count = cnt;
			uData[u]->ratings = (Rating*) uData[u] + sizeof(uData);
		}
		uData[u]->ratings[i].id = it->second.id;
		uData[u]->ratings[i].value = it->second.value;
		// printf("%d u%d r%f\n", &(mData[m]->ratings[i]), it->second.id, it->second.value);
	}
	// for (int u = 0; u < U; u++) {
	// 	if (uData[u]) {
	// 		printf("u%d\n", uData[u]->count);
	// 		for (int i = 0; i < uData[u]->count; i++) {
	// 			printf("m%d %f ", uData[u]->ratings[i].id, uData[u]->ratings[i].value);
	// 		}
	// 		printf("\n");
	// 	}
	// }
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
void calculateMagnitude(Profile1** mData, int M) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  	unsigned int stride = blockDim.x * gridDim.x;

	for(unsigned int m = index; m < M; m += stride) {
		if (!mData[m]) continue;
		/* Calculate Mean */
		double mean = 0;
		for (int i = 0; i < mData[m]->count; i++) {
			mean += mData[m]->ratings[i].value;
		}
		mean /= (double) mData[m]->count;

		/* Calculate Magnitude */
		mData[m]->magnitude = 0;
		for (int i = 0; i < mData[m]->count; i++) {
			mData[m]->ratings[i].value -= mean;
			mData[m]->magnitude += mData[m]->ratings[i].value * mData[m]->ratings[i].value;
		}
		mData[m]->magnitude = sqrt(mData[m]->magnitude);
	}
}

__global__
void calculateSimilarity(Profile1** data, double* similarities, int M) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int unit = M / stride + 1;
	
	for(unsigned int m = index; m < M; m += stride) {
		if (!data[m]) continue;
	// for(unsigned int m = 0; m < M; m ++) {
		int cnt = data[m]->count;
		Rating *ratings = data[m]->ratings;
		for(int n = 0; n < M; n++) {
			if (!data[n]) continue;
			if (((n + m) % 2 && n < m) || ((n + m) % 2 == 0 && n > m)) {
				double mag_prod = data[m]->magnitude * data[n]->magnitude;
				if (mag_prod == 0) {
					similarities[m*M + n] = 0;
					similarities[n*M + m] = 0;
					// printf("%d %d: %f %f\n", m, n, data[m]->magnitude, data[n]->magnitude);
				}
				else {
					int i = 0, j = 0;
					double dot_prod = 0;

					// printf("\n%d %d\n", m, n);
					while (i < cnt && j < data[n]->count) {
						while (i < cnt && ratings[i].id < data[n]->ratings[j].id) i++;
						while (j < data[n]->count && ratings[i].id > data[n]->ratings[j].id) j++;
						if (ratings[i].id == data[n]->ratings[j].id) {
							dot_prod += ratings[i].value * data[n]->ratings[j].value;
							// printf("i%d j%d, %d %d\n", i, j, ratings[i].id, data[n]->ratings[j].id);
							i++, j++;
						}
					}
					similarities[m*M + n] = dot_prod / mag_prod;
					similarities[n*M + m] = dot_prod / mag_prod;
					// printf("%d %d: %f / %f = %f\n", m, n, dot_prod, mag_prod, similarities[m*M + n]);
				}
			}
		}
	}
}

/* Sync movieId to match index of similarities matrix */
template <class T>
__device__
void syncIndex(T* uData, int *movieList, int M) {
	int i = 0, m = 0;
	while (i < uData->count && m < M) {
		if(uData->ratings[i].id == movieList[m]) {
			// uData[u]->ratings[i].id;
			uData->ratings[i].id = m;
			// printf("%d %d, %d %d\n", i, uData[u]->count, uData[u]->ratings[i].id, movieList[m]);
			i++, m++;
		}
		while(i < uData->count && uData->ratings[i].id < movieList[m]) i++;
		while(m < M && uData->ratings[i].id > movieList[m]) m++;
	}
}

__global__ 
void predictRatings(double *predictions, bool mode, Profile2** data, double *similarities, int *movieList, int U, int M) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int unit = U / stride + 1;

	/* Sync movieId to match index of similarities matrix */
	for(unsigned int u = index + mode; u < U; u += stride) {
	// for(unsigned int u = 1; u <= U; u++) {
		if (!data[u]) continue;
		if (mode == 1) syncIndex<Profile2>(data[u], movieList, M);
		int i = 0, m = 0;
		// printf("cnt %d\n", data[u]->count);
		// for (int i = 0; i < data[u]->count; i++) {
		// 	printf("m%d %f ", data[u]->ratings[i].id, data[u]->ratings[i].value);
		// }
		// printf("\n");

		/* Predict rating */
		m = 0;
		for(i = 0; i <= data[u]->count; i++) {
			// printf("%d %d %d\n", u, i, data[u]->count);
			int end = (i == data[u]->count)? M : data[u]->ratings[i].id;
			for (; m < end; m++) {
				// printf("**%d %d %d\n", u, m, movieList[data[u]->ratings[i].id]);
				double sum_sim = 0, sum_rating = 0;
				for (int n = 0; n < data[u]->count; n++) {
					Rating n_r = data[u]->ratings[n];
					
					// printf("\n+%d, %d, %d", u, m, movieId[n]);
					double sim = similarities[m*M + n_r.id];
					if (sim > 0) {
						// printf("%d %d %f\n", u, m, n, n_r.value);
						sum_sim += sim;
						sum_rating += sim * data[u]->ratings[n].value;
					}
				}
				if (sum_sim > 0) {
					double pred = sum_rating / sum_sim;
					if (predict) {
						if (mode == 1) printf("%d %d %f\n", u, movieList[m], pred);
						else printf("%d %d %f\n", m, movieList[u], pred);
					}
				}
				// predictions[u*U + movieId[m]] = (sum_sim == 0)? 0 : sum_rating / sum_sim;
			}
			// if (m < M) predictions[u*U + movieId[m]] = -1;
			// printf("%d %d\n", u, m);
			m++;
		}
	}
}

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "Please provide input file\n");
		return 1;
	}
	bool mode = strcmp(argv[2], "user");
	clock_t start, end;
	double elapsed = 0, totalTime = 0;
	int U, M, N;
	
	/* Input Processing */
	std::ifstream fin(argv[1]);
	if (mode == 1) {
		fin >> U >> M >> N;
		U++;
	}
	else {
		fin >> M >> U >> N;
		M++;
	}
	printGPUMemmory();
	
    start = clock();
	int* movieList;
	Profile1 **data1;
	checkCuda( cudaMallocManaged(&movieList, sizeof(int) * M) );
    checkCuda( cudaMallocManaged(&data1, sizeof(Profile1*) * M) );	
	if (mode == 1) mapMovie<Profile1>(argv[1], data1, movieList);	
	else mapUser<Profile1>(argv[1], data1);
	end = clock();
	elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
	// totalTime += elapsed;
	if (eval) {
		printf("Input Processing (ms): %f\n", elapsed);
		printGPUMemmory();
		if (predict) {
			// printData(count1, rating1, M);
		}
	}
	
	/* Calculate Magnitude */
    start = clock();
	calculateMagnitude<<<M, 1>>>(data1, M);
	checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
	end = clock();
    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    totalTime += elapsed;
	if (eval) {
		printf("Calculate Magnitude [1] (ms): %f\n", elapsed);
		printGPUMemmory();
		if (predict) {
			for (int i = 0; i < M; i++) {
				// printf("%d, %f\n", i, magnitudes[i]);
			}
		}
	}
	// for (int i = 0; i < M; i++) {
	// 	printf("%d, %f\n", movieList[i], data1[i]->magnitude);
	// }
	
	
	/* Calculate Similarities */
	start = clock();	
	double* similarities;
	checkCuda( cudaMallocManaged(&similarities, sizeof(double) * M* (M + M)) );
	
	calculateSimilarity<<<M, 1>>>(data1, similarities, M);	
	// calculateSimilarity(data1, similarities, M);	
	checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
	end = clock();
    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    totalTime += elapsed;
    if (eval) {
		printf("Calculate Similarities [1] (ms): %f\n", elapsed);
		printGPUMemmory();
		if (predict) {
			for (int i = 1 - mode; i < M; i++) {
				for (int j = i+1; j < M; j++) {
					if (mode == 1) printf("%d %d, %f\n", movieList[i], movieList[j], similarities[i*M+j]);
					else printf("%d %d, %f\n", i, j, similarities[i*M+j]);
				}
			}
		}
	}
	// for (int i = 1 - mode; i < M; i++) {
	// 	for (int j = i+1; j < M; j++) {
	// 		if (mode == 1) printf("%d %d, %f\n", movieList[i], movieList[j], similarities[i*M+j]);
	// 		else printf("%d %d, %f\n", i, j, similarities[i*M+j]);
	// 	}
	// }

	// // checkCuda( cudaFree(count) );
	// // checkCuda( cudaFree(similarities) );
	
	// /* Predict Ratings */
	cudaSetDevice(1);
	printGPUMemmory();
	start = clock();

	int* ratings;
	int* count;
	Profile2 **data2;	
    checkCuda( cudaMallocManaged(&data2, sizeof(Profile2*) * U) );	
	if (mode == 1) mapUser<Profile2>(argv[1], data2);	
	else mapMovie<Profile2>(argv[1], data2, movieList);
	if (eval && predict) {
		printGPUMemmory();
		// printData(count2, rating2, U);
	}	

	double* predictions;
    // checkCuda( cudaMallocManaged(&predictions, sizeof(double) * N * M) );

	// printf("Memory required:\t%llu\n", sizeof(Profile2) * U);
	// printf("Memory required:\t%llu\n", sizeof(double) * M * M + sizeof(Profile2) * U + sizeof(Rating) * N);
	predictRatings<<<U, 1>>>(predictions, mode, data2, similarities, movieList, U, M);
	// predictRatings(predictions, data2, similarities, movieList, U, M); 
	checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
	end = clock();
    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    totalTime += elapsed;
    if (eval) {
		// if (predict) printData(count2, rating2, U);
		printGPUMemmory();
		printf("Predict Ratings [1] (ms): %f\n", elapsed);
    	printf("Total time (ms): %f\n", totalTime);
	}	
}
