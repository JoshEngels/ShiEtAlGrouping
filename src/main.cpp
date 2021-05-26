#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm> 
#include <MurmurHash3.h>
#include <limits>
#include <queue>
#include <unordered_set>

using namespace std;
using Sparse_Vector = vector<pair<int, float>>;
using Dataset = vector<Sparse_Vector>;


// Some memory fragmentation but it's fine, not the performance important part
Dataset readSparse(string fileName) {
  
  Dataset result;

  ifstream file(fileName);
  string str;

  int currentVectorNum = 0;
  while (getline(file, str)) {
    Sparse_Vector currentVector;

    // Constructs an istringstream object iss with a copy of str as content.
    istringstream iss(str);
    
    // Removes label.
    string sub;
    iss >> sub;

    // Loop through and remove key value pairs
    do {
      string sub;
      iss >> sub;
      size_t pos = sub.find_first_of(":");
      if (pos == string::npos) {
        continue;
      }
      float val = stof(sub.substr(pos + 1, (str.length() - 1 - pos)));
      pos = stoi(sub.substr(0, pos));
      currentVector.emplace_back(pos, val);
    } while (iss);

    result.push_back(currentVector);
    currentVectorNum++;
  }

  return result;
}

vector<int> readGroundTruth(string fileName) {
  
  vector<int> result;
  ifstream file(fileName);
  string str, buf;

  getline(file, str);
  stringstream ss(str);

  while (ss >> buf) {
    result.push_back(stoi(buf));
  }

  return result;

}

float compute_mean_recall(vector<int> results, int k, vector<int> gtruth, int k_gtruth, int denominator, int numerator, int numQueries){
    float mean_recall = 0;

    for (int n = 0; n < numQueries; n++){
        int n_correct = 0;

        for (int i = 0; i < denominator; i++){
            for (int j = 0; j < numerator; j++){
                if (results[k*n + i] == gtruth[k_gtruth*n + j]){
                    n_correct++;
                }
            }
        }
        float recall = (float)(n_correct)/numerator;
        mean_recall = mean_recall + recall;
    }

    return mean_recall / numQueries; 
}

float compute_mean_precision(vector<int> results, int k, vector<int> gtruth, int k_gtruth, int denominator, int numerator, int numQueries){

    float mean_precision = 0;

    for (int n = 0; n < numQueries; n++){
        int n_correct = 0;

        for (int i = 0; i < denominator; i++){
            for (int j = 0; j < numerator; j++){
                if (results[k*n + i] == gtruth[k_gtruth*n + j]){
                    n_correct++;
                }
            }
        }
        float precision = (float)(n_correct)/denominator;
        mean_precision = mean_precision + precision;
    }

    return mean_precision / numQueries; 
}

float *project(Dataset allData, size_t newDimension) {
  float *newData = (float *)calloc(newDimension * allData.size(), sizeof(float));
#pragma omp parallel for
  for (size_t dataID = 0; dataID < allData.size(); dataID++) {
    for (pair<int, float> sparse : allData[dataID]) {
      for (size_t d = 0; d < newDimension; d++) {
        int32_t result;
        MurmurHash3_x86_32(&sparse.first, sizeof(int), d, &result);
        float diff = (float)result * sparse.second / (float)INT32_MAX;
        newData[newDimension * dataID + d] += diff;
      }
    }      
  }

  // Unitize vectors  
  float *magnitudes = (float *)calloc(allData.size(), sizeof(float));
#pragma omp parallel for
  for (size_t dataID = 0; dataID < allData.size(); dataID++) {
    for (size_t d = 0; d < newDimension; d++) {
      magnitudes[dataID] += pow(newData[newDimension * dataID + d], 2);
    }
  }
  for (size_t i = 0; i < allData.size(); i++) {
    magnitudes[i] = pow(magnitudes[i], 0.5);
  }
#pragma omp parallel for
  for (size_t dataID = 0; dataID < allData.size(); dataID++) {
    for (size_t d = 0; d < newDimension; d++) {
      newData[newDimension * dataID + d] /= magnitudes[dataID];
    }
  }
  free(magnitudes);

  return newData;
}

// TODO: Add back propogation and related components
pair<vector<int>, float> doQueries(float *data, size_t dimension, size_t numQueries, size_t numData, size_t k, size_t reps, size_t groupsPerRep, size_t numBatches, size_t batchSize) {
  
  vector<int> results(numQueries * k);

  // Group i will be all indices that equal i % totalGroups
  vector<int> groupIndices;
  for (size_t i = 0; i < numData; i++) {
    groupIndices.push_back(i);
  }

  // Groups numbered 0 ... totalGroups - 1
  size_t totalGroups = reps * groupsPerRep;

  // Entry i of memory vector for group j is j * totalGroups + i
  float *groupVectors = (float *)calloc(totalGroups * dimension, sizeof(float));
  // The rep'th group of vector j is j * reps + rep
  int *dataToGroups = (int *)calloc(numData * reps, sizeof(int));
  // Vectors for group i are in groupsToData[i]
  vector<vector<int>> groupsToData(totalGroups);

  cout << "Starting indexing " << endl;

  for (size_t rep = 0; rep < reps; rep++) {

    // Get random group assignment
    random_shuffle(groupIndices.begin(), groupIndices.end());
#pragma omp parallel for
    for (size_t groupInRep = 0; groupInRep < groupsPerRep; groupInRep++) {
      
      size_t group = rep * groupsPerRep + groupInRep;
      
      for (size_t dataID = groupInRep; dataID < numData; dataID += groupsPerRep) {
        dataToGroups[dataID * reps + rep] = group;
        groupsToData[group].push_back(dataID);
        for (size_t d = 0; d < dimension; d++) {
          groupVectors[group * dimension + d] += data[(dataID + numQueries) * dimension + d];
        }    
      }
    }
  }
  
  // Do actual queries
  cout << "Querying" << endl;
  auto start = chrono::high_resolution_clock::now();
// #pragma omp parallel for
  for (size_t query = 0; query < numQueries; query++) {
    // Find the dot product of all memory vectors
    float* dots = (float *)calloc(totalGroups, sizeof(float));
    for (size_t group = 0; group < totalGroups; group++) {
      for (size_t d = 0; d < dimension; d++) {
        dots[group] += groupVectors[group * dimension + d] * data[query * dimension + d];
      }
    }

    priority_queue<pair<float, int>> best;
    unordered_set<int> used;
    for (size_t batchNum = 0; batchNum < numBatches; batchNum++) {
      vector<pair<double, int>> scores;
      for (size_t i = 0; i < numData; i++) {
        if (used.count(i)) {
          continue;
        }
        double score = 0;
        for (size_t rep = 0; rep < reps; rep++) {
          score += dots[dataToGroups[i * reps + rep]];
        }
        scores.emplace_back(-score, i);
      }

      sort(scores.begin(), scores.end());

      for (size_t n = 0; n < batchSize; n++) {
        double dot = 0;
        used.insert(scores[n].second);
        for (size_t d = 0; d < dimension; d++) {
          dot += data[(scores[n].second + numQueries) * dimension + d] * data[query * dimension + d];
        }
        if (best.size() < k || -dot < best.top().first) {
          best.emplace(-dot, scores[n].second);
          if (best.size() > k) {
            best.pop();
          }
        }
        for (size_t rep = 0; rep < reps; rep++) {
          dots[dataToGroups[reps * scores[n].second]] -= dot;
        }
      }
    }
    for (size_t n = 0; n < k; n++) {
      results.at((query + 1) * k - n - 1) = best.top().second;
      best.pop();
    }
    free(dots);
  }
  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(stop-start);
  float mean_latency = (float)(duration.count()) / numQueries;  

  free(groupVectors);
  free(dataToGroups);

  return make_pair(results, mean_latency);
}

pair<vector<int>, float> bruteForce(float *data, size_t dimension, size_t numQueries, size_t numData, size_t k) {

  vector<int> results(numQueries * k);
#pragma omp parallel for
  for (size_t q = 0; q < numQueries; q++) {
    vector<pair<float, int>> dots;
    for (size_t dataID = 0; dataID < numData; dataID++) {
      float dot = 0;
      for (size_t d = 0; d < dimension; d++) {
        dot += data[q * dimension + d] * data[(numQueries + dataID) * dimension + d];
      }
      dots.emplace_back(-dot, dataID);
    }
    sort(dots.begin(), dots.end());
    for (size_t n = 0; n < k; n++) {
      results.at(q * k + n) = dots.at(n).second;
    }
  }

  return make_pair(results, 0);
}


int main(int argc, char **argv){

    if (argc < 11){
        clog << "Usage: " << endl;
        clog << "run <data_file> <num_queries> <gruth_file> <k> <k_gtruth> <groups_per_rep> <reps> <batch_size> <num_batches> <new_dimension>" << endl;
        return -1; 
    }

    size_t numQueries = stoi(argv[2]);
    size_t k = stoi(argv[4]);
    size_t kGtruth = stoi(argv[5]);
    size_t groupsPerRep = stoi(argv[6]);
    size_t groupReps = stoi(argv[7]);
    size_t batchSize = stoi(argv[8]);
    size_t numBatches = stoi(argv[9]);
    size_t newDimension = stoi(argv[10]);

    Dataset allData = readSparse(argv[1]);
    vector<int> gtruth = readGroundTruth(argv[3]);

    float *denseData = project(allData, newDimension);
    auto results = doQueries(denseData, newDimension, numQueries, allData.size() - numQueries, k, groupReps, groupsPerRep, numBatches, batchSize);

    // auto results = bruteForce(denseData, newDimension, numQueries, allData.size() - numQueries, k);

    size_t numerators[]{1, 10, 100};
    for (size_t numerator : numerators) {
        for (size_t denominator = numerator; denominator <= k; denominator++) {
            float mean_recall = compute_mean_recall(results.first, k, gtruth, kGtruth, denominator, numerator, numQueries);
            float mean_precision = compute_mean_precision(results.first, k, gtruth, kGtruth, denominator, numerator, numQueries);
            cout<<"R"<<numerator<<"@"<<denominator<<" "<<results.second<<","<<mean_recall<<","<<mean_precision<<endl;
        }
    }
}