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

pair<vector<int>, float> doQueries(size_t topK, Dataset allData, size_t numQueries, size_t groupsPerRep, size_t reps, size_t dataDim) {
  
  vector<int> results;
  size_t numData = allData.size() - numQueries;

  vector<int> groupIndices;
  for (size_t i = 0; i < numData; i++) {
    groupIndices.push_back(i % groupsPerRep);
  }

  // Groups numbered 0 ... totalGroups - 1
  size_t totalGroups = reps * groupsPerRep;


  // Entry i of memory vector for group j is i * totalGroups + j
  float *groupVectors = (float *)calloc(totalGroups * dataDim, sizeof(float));
  // The rep'th group of vector j is j * reps + rep
  int *dataToGroups = (int *)calloc(numData * reps, sizeof(int));
  // Vectors for group i are in groupsToData[i]
  vector<vector<int>> groupsToData(totalGroups);

  cout << "Starting indexing " << endl;

  for (size_t rep = 0; rep < reps; rep++) {
    
    // Get random group assignment
    random_shuffle(groupIndices.begin(), groupIndices.end());

    for (size_t dataID = 0; dataID < numData; dataID++) {
      size_t group = groupIndices[dataID] + rep * groupsPerRep;
      dataToGroups[dataID * reps + rep] = group;
      groupsToData[group].push_back(dataID); 

      // Add vector to memory vectors
      for (pair<int, float> sparse : allData[numQueries + dataID]) {
        groupVectors[sparse.first * totalGroups + group] += sparse.second;
      }
    }
  }

  // Unitize memory vectors
  cout << "Unitizing" << endl;
  float *magnitudes = (float *)calloc(totalGroups, sizeof(float));
  for (size_t i = 0; i < totalGroups * dataDim; i++) {
    magnitudes[i % totalGroups] += pow(groupVectors[i], 2);
  }
  for (size_t i = 0; i < totalGroups; i++) {
    magnitudes[i] = pow(magnitudes[i], 0.5);
  }
  for (size_t i = 0; i < totalGroups * dataDim; i++) {
    groupVectors[i] /= magnitudes[i % totalGroups];
  }  
  free(magnitudes);
  
  // Do actual queries
  cout << "Querying" << endl;
  auto start = chrono::high_resolution_clock::now();
  for (size_t query = 0; query < numQueries; query++) {

    float* dots = (float *)calloc(totalGroups, sizeof(float));

    // Find the dot product of all memory vectors
    for (pair<int, float> sparse : allData[query]) {
      for (size_t group = 0; group < totalGroups; group++) {
        dots[group] += groupVectors[sparse.first * totalGroups + group] * sparse.second;
      }
    }


    vector<pair<double, int>> scores;
    for (size_t i = 0; i < numData; i++) {
      double score = 0;
      for (size_t rep = 0; rep < reps; rep++) {
        score += dots[dataToGroups[i * reps + rep]];
      }
      scores.emplace_back(-score, i);
    }
    free(dots);

    sort(scores.begin(), scores.end());

    for (size_t k = 0; k < topK; k++) {
      results.push_back(scores[k].second);
      cout << scores[k].second << endl;
    }
  }
  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(stop-start);
  float mean_latency = (float)(duration.count()) / numQueries;  

  free(groupVectors);
  free(dataToGroups);

  return make_pair(results, mean_latency);
}


int main(int argc, char **argv){

    if (argc < 8){
        clog << "Usage: " << endl;
        clog << "run <data_file> <num_queries> <gruth_file> <k> <k_gtruth> <groups_per_rep> <reps> <data_dim>" << endl;
        return -1; 
    }

    size_t numQueries = stoi(argv[2]);
    size_t k = stoi(argv[4]);
    size_t kGtruth = stoi(argv[5]);
    size_t groupsPerRep = stoi(argv[6]);
    size_t groupReps = stoi(argv[7]);
    size_t dataDim = stoi(argv[8]);

    Dataset allData = readSparse(argv[1]);
    vector<int> gtruth = readGroundTruth(argv[3]);


    cout << "Data read in " << allData.size() << endl;

    // TODO: Change to loop over lots of numbers of groups
    auto results = doQueries(k, allData, numQueries, groupsPerRep, groupReps, dataDim);

    size_t numerators[]{1, 10, 100};
    for (size_t numerator : numerators) {
        for (size_t denominator = numerator; denominator <= k; denominator++) {
            float mean_recall = compute_mean_recall(results.first, k, gtruth, kGtruth, denominator, numerator, numQueries);
            float mean_precision = compute_mean_precision(results.first, k, gtruth, kGtruth, denominator, numerator, numQueries);
            cout<<"R"<<numerator<<"@"<<denominator<<" "<<results.second<<","<<mean_recall<<","<<mean_precision<<endl;
        }
    }
}