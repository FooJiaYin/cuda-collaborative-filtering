#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;

// #define show_output

struct Rating {
	int id;
	double value;
};

void printData(int* count, Rating* ratings, int M) {
	int i = 0, j = 0;
	for (i = 0; i < M; i++) {
		printf("%d: ", i);
		for (; j < count[i]; j++) {
			printf("| %d: %f ", ratings[j].id, ratings[j].value);
		}
		printf("\n");
	}
}

void readInput(const char* filename, bool mode, multimap<int, Rating>& ratingsMap) { 
	std::ifstream fin(filename);

	int U, M, N;
	fin >> U >> M >> N;
	
	for (int i = 0; i < N; i++) {
		int k1 = 0, k2 = 0;
		double v = 0;
		if (mode == 0) fin >> k1 >> k2 >> v;
		else fin >> k2 >> k1 >> v;
		Rating r;
		r.id = k2;
		r.value = v;
		ratingsMap.insert({k1, r});
	}
}

void mapMovie(const char* filename, Rating* ratings, int* count, int* movieList) {
	multimap<int, Rating> ratingsMap;
	readInput(filename, 1, ratingsMap);

	int i = 0, m = -1;	
	movieList[0] = 0; 	
    // printf("%d\n", ratingsMap.size());
	for (auto it = ratingsMap.begin(); it != ratingsMap.end(); it++, i++) {
		if (m == -1 || it->first != movieList[m]) {
			if(m >= 0) count[m] = i;
			// printf("| cnt: %d\n", count[m]);
            if (count[m] == ratingsMap.size() -1) break;
			m++;
			movieList[m] = it->first;
			// printf("%d: ", movieList[m]);
		}
		ratings[i].id = it->second.id;
		ratings[i].value = it->second.value;
		// printf("| %d: %f ", ratings[i].id, ratings[i].value);
	}
	count[m] = i;
	// printf("\n\n\n");
}

void mapUser(const char* filename, Rating* ratings, int* count) {
	multimap<int, Rating> ratingsMap;
	readInput(filename, 0, ratingsMap);

	int i = 0, m = -1;	
	count[0] = 0;
	for (auto it = ratingsMap.begin(); it != ratingsMap.end(); it++, i++) {
		if (m == -1 || it->first != m) {
			if(m >= 0) count[m] = i;
			// printf("| cnt: %d\n", count[m]);
			m = it->first;
			// printf("%d: ", m);
		}
		ratings[i].id = it->second.id;
		ratings[i].value = it->second.value;
		// printf("| %d: %f ", ratings[i].id, ratings[i].value);
	}
	count[m] = i;
	// printf("\n");
}

void calculateMagnitude(Rating* ratings, int* count, double* magnitudes, int M) {
	for(unsigned int m = 0; m < M; m ++) {
		if (count[m] == 0) continue;
		int cnt = count[m];
		int start = (m > 0) ? count[m-1] : 0;

		/* Calculate Mean */
		double mean = 0;
		for (int i = start; i < cnt; i++) {
			mean += ratings[i].value;
		}
		mean /= (double) cnt - start;

		/* Calculate Magnitude */
		double mag = 0;
		for (int i = start; i < cnt; i++) {
			ratings[i].value -= mean;
			mag += ratings[i].value * ratings[i].value;
		}
		magnitudes[m] = sqrt(mag);
	}
}

void calculateSimilarity(Rating* ratings, int* count, double* magnitudes, double* similarities, int M) {
	for(unsigned int m = 0; m < M; m++) {
		if (count[m] == 0) continue;
		int cnt = count[m];
		double mag = magnitudes[m];

		for(int n = m + 1; n < M; n++) {
			if (count[n] == 0) continue;
            double mag_prod = mag * magnitudes[n];
            if (mag_prod == 0) {
                similarities[m*M + n] = 0;
                similarities[n*M + m] = 0;
            }
            else {
                int i = (m > 0)? count[m-1] : 0;
                int j = (n > 0)? count[n-1] : 0;
                int cnt_2 = count[n];
                double dot_prod = 0;

                while (i < cnt && j < cnt_2) {
                    while (i < cnt && ratings[i].id < ratings[j].id) i++;
                    while (j < cnt_2 && ratings[i].id > ratings[j].id) j++;
                    if (ratings[i].id == ratings[j].id) {
                        dot_prod += ratings[i].value * ratings[j].value;
                        i++, j++;
                    }
                }
                similarities[m*M + n] = dot_prod / mag_prod;
                similarities[n*M + m] = dot_prod / mag_prod;
            }
		}
	}
}

/* Sync movieId to match index of similarities matrix */
void syncIndex(int start, int count, Rating* ratings, int *movieList, int M) {
	int i = start, m = 0;
	while (i < count && m < M) {
		// printf("%d %d %d, %d %d %d\n", i, start, count, m, ratings[i].id, movieList[m]);
		if(ratings[i].id == movieList[m]) {
			// uData[u]->ratings[i].id;
			// printf("**%d %d %d, %d %d %d\n", i, start, count, m, ratings[i].id, movieList[m]);
			ratings[i].id = m;
			i++, m++;
		}
		while(i < count && ratings[i].id < movieList[m]) i++;
		while(m < M && ratings[i].id > movieList[m]) m++;
	}
}

void predictRatings(double *predictions, bool mode, int *count, Rating *ratings, double *similarities, int *movieList, int U, int M) {
	
    /* Sync movieId to match index of similarities matrix */
	for(unsigned int u = mode; u < U; u++) {
		if (count[u] == 0) continue;
		int start = (u > 0)? count[u-1] : 0;
		int cnt = count[u];
		if (mode == 1) syncIndex(start, cnt, ratings, movieList, M);

		/* Predict rating */
		int i = start, m = 0;
		for(i = start; i <= cnt; i++) {
			int end = (i == cnt)? M : ratings[i].id;
			for (; m < end; m++) {
				double sum_sim = 0, sum_rating = 0;
				for (int n = start; n < cnt; n++) {
					double sim = similarities[m*M + ratings[n].id];
					if (sim > 0) {
						sum_sim += sim;
						sum_rating += sim * ratings[n].value;
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
	// multimap<int, pair<int, double>> m_u_ratings;
	// read_input(argv[1], argv[2], U, M, N, m_u_ratings);

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
	if (predict && eval) printf("M: %d U: %d\n", M, U);

	int *movieList = (int *) malloc(sizeof(int) * M);
	int *count1 = (int *) malloc(sizeof(int) * M);
	double *magnitudes = (double *) malloc(sizeof(double) * M);
	Rating *rating1 = (Rating *) malloc(sizeof(Rating) * N);

	start = clock();
	if (mode == 1) mapMovie(argv[1], rating1, count1, movieList);	
	else mapUser(argv[1], rating1, count1);
	end = clock();
	elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
	totalTime += elapsed;
    if (eval) {
		printf("Input Processing (ms): %f\n", elapsed);
		if (predict) {
			printData(count1, rating1, M);
		}
	}
	
    // // split(movieList, count1, userId, u_ratings, m_u_ratings, N);

	/* Calculate Magnitude */
    start = clock();
	calculateMagnitude(rating1, count1, magnitudes, M);
	end = clock();
    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
	totalTime += elapsed;
	if (eval) {
		printf("Calculate Magnitude (ms): %f\n", elapsed);
		if (predict) {
			for (int i = 0; i < M; i++) {
				printf("%d, %f\n", i, magnitudes[i]);
			}
		}
	}
	
	/* Calculate Similarities */
	start = clock();
	double* similarities = (double *) malloc(sizeof(double) * M* (M + M));
	calculateSimilarity(rating1, count1, magnitudes, similarities, M);	
	end = clock();
    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
	totalTime += elapsed;
    if (eval) {
		printf("Calculate Similarities (ms): %f\n", elapsed);
		if (predict) {
			for (int i = 1 - mode; i < M; i++) {
				for (int j = i+1; j < M; j++) {
					if (mode == 1) printf("%d %d, %f\n", movieList[i], movieList[j], similarities[i*M+j]);
					else printf("%d %d, %f\n", i, j, similarities[i*M+j]);
				}
			}
		}
	}
    free(rating1);
    free(count1);
    free(magnitudes);
	
	/* Predict Ratings */
	Rating* rating2 = (Rating *) malloc(sizeof(Rating) * N);
	int* count2 = (int *) malloc(sizeof(int) * U);
	double* predictions;
	if (mode == 1) mapUser(argv[1], rating2, count2);
	else mapMovie(argv[1], rating2, count2, movieList);
	if (eval && predict) {
		printData(count2, rating2, U);
	}
	
	start = clock();
	predictRatings(predictions, mode, count2, rating2, similarities, movieList, U, M);
	end = clock();
    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
	totalTime += elapsed;
    if (eval) {
		if (predict) printData(count2, rating2, U);
		printf("Predict Ratings (ms): %f\n", elapsed);
    	printf("Total time (ms): %f\n", totalTime);
	} 
}

