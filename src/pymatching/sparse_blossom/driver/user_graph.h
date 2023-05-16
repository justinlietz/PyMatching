// Copyright 2022 PyMatching Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PYMATCHING2_USER_GRAPH_H
#define PYMATCHING2_USER_GRAPH_H

#include <cmath>
#include <list>
#include <set>
#include <vector>

#include "pymatching/rand/rand_gen.h"
#include "pymatching/sparse_blossom/driver/io.h"
#include "pymatching/sparse_blossom/ints.h"

namespace pm {

  class CSCCheckMatrix {
    public:
      std::vector<uint8_t> data;
      std::vector<uint64_t> indices;
      std::vector<uint64_t> indptr;
      size_t num_rows;
      size_t num_cols;

      CSCCheckMatrix(const std::vector<uint8_t> &data_in,
          const std::vector<uint64_t> &indices_in,
          const std::vector<uint64_t> &indptr_in,
          size_t num_rows_in,
          size_t num_cols_in):
        data(data_in), indices(indices_in), indptr(indptr_in), num_rows(num_rows_in), num_cols(num_cols_in){}
      CSCCheckMatrix(const std::vector<std::vector<uint8_t>> &vectors_in):
        num_rows(vectors_in.size()), num_cols(vectors_in[0].size()){
          indptr.push_back(data.size());
          for(size_t c = 0; c < num_cols; ++c){
            for(size_t r = 0; r < num_rows; ++r){
              if( vectors_in[r][c] != 0 ){
                data.push_back(vectors_in[r][c]);
                indices.push_back(r);
              }
            }
            indptr.push_back(data.size());
          }
        }

      uint8_t get(size_t row, size_t col) const{
        for(size_t idx = indptr[col]; idx < indptr[col+1]; ++idx){
          if( row == indices[idx] ){
            return data[idx];
          }
        }
        return 0;
      }

      void print_dense() const{
        std::cout << "num_rows: " << this->num_rows << "\n";
        std::cout << "num_cols: " << this->num_cols << "\n";
        std::cout << "data.size(): " << this->data.size() << "\n";
        std::cout << "indices.size(): " << this->indices.size() << "\n";
        std::cout << "indptr.size(): " << this->indptr.size() << "\n";
        for(size_t r = 0; r < this->num_rows; ++r){
          for(size_t c = 0; c < this->num_cols; ++c){
            // "+" helps print uint8_t as printable numerical value
            std::cout << +this->get(r,c) << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";
      }
  };

struct UserEdge {
    size_t node1;
    size_t node2;
    std::vector<size_t> observable_indices;  /// The indices of the observables crossed along this edge
    double weight;                           /// The weight of the edge to this neighboring node
    double error_probability;                /// The error probability associated with this node
};

struct UserNeighbor {
    std::list<UserEdge>::iterator edge_it;
    uint8_t pos{};  // The position of the neighboring node in the edge (either 0 if node1, or 1 if node2)
};

class UserNode {
   public:
    UserNode();
    size_t index_of_neighbor(size_t node) const;
    std::vector<UserNeighbor> neighbors;  /// The node's neighbors.
    bool is_boundary;
};

const pm::weight_int MAX_USER_EDGE_WEIGHT = NUM_DISTINCT_WEIGHTS - 1;

enum MERGE_STRATEGY : uint8_t { DISALLOW, INDEPENDENT, SMALLEST_WEIGHT, KEEP_ORIGINAL, REPLACE };

class UserGraph {
   public:
    std::vector<UserNode> nodes;
    std::list<UserEdge> edges;
    std::set<size_t> boundary_nodes;

    UserGraph();
    explicit UserGraph(size_t num_nodes);
    UserGraph(size_t num_nodes, size_t num_observables);
    void merge_edge_or_boundary_edge(
        size_t node,
        size_t neighbor_index,
        const std::vector<size_t>& parallel_observables,
        double parallel_weight,
        double parallel_error_probability,
        MERGE_STRATEGY merge_strategy = DISALLOW);
    void add_or_merge_edge(
        size_t node1,
        size_t node2,
        const std::vector<size_t>& observables,
        double weight,
        double error_probability,
        MERGE_STRATEGY merge_strategy = DISALLOW);
    void add_or_merge_boundary_edge(
        size_t node,
        const std::vector<size_t>& observables,
        double weight,
        double error_probability,
        MERGE_STRATEGY merge_strategy = DISALLOW);
    bool has_edge(size_t node1, size_t node2);
    bool has_boundary_edge(size_t node);
    void set_boundary(const std::set<size_t>& boundary);
    std::set<size_t> get_boundary();
    size_t get_num_observables();
    void set_min_num_observables(size_t num_observables);
    size_t get_num_nodes();
    size_t get_num_detectors();
    size_t get_num_edges();
    bool is_boundary_node(size_t node_id);
    void add_noise(uint8_t* error_arr, uint8_t* syndrome_arr) const;
    bool all_edges_have_error_probabilities();
    double max_abs_weight();
    double get_edge_weight_normalising_constant(size_t max_num_distinct_weights);
    template <typename EdgeCallable, typename BoundaryEdgeCallable>
    double iter_discretized_edges(
        pm::weight_int num_distinct_weights,
        const EdgeCallable& edge_func,
        const BoundaryEdgeCallable& boundary_edge_func);
    pm::MatchingGraph to_matching_graph(pm::weight_int num_distinct_weights);
    pm::SearchGraph to_search_graph(pm::weight_int num_distinct_weights);
    pm::Mwpm to_mwpm(pm::weight_int num_distinct_weights, bool ensure_search_graph_included);
    void update_mwpm();
    Mwpm& get_mwpm();
    Mwpm& get_mwpm_with_search_graph();
    void handle_dem_instruction(double p, const std::vector<size_t>& detectors, const std::vector<size_t>& observables);
    void get_nodes_on_shortest_path_from_source(size_t src, size_t dst, std::vector<size_t>& out_nodes);

   private:
    pm::Mwpm _mwpm;
    size_t _num_observables;
    bool _mwpm_needs_updating;
    bool _all_edges_have_error_probabilities;
};

template <typename EdgeCallable, typename BoundaryEdgeCallable>
inline double UserGraph::iter_discretized_edges(
    pm::weight_int num_distinct_weights,
    const EdgeCallable& edge_func,
    const BoundaryEdgeCallable& boundary_edge_func) {
    pm::MatchingGraph matching_graph(nodes.size(), _num_observables);
    double normalising_constant = get_edge_weight_normalising_constant(num_distinct_weights);

    for (auto& e : edges) {
        pm::signed_weight_int w = (pm::signed_weight_int)round(e.weight * normalising_constant);
        // Extremely important!
        // If all edge weights are even integers, then all collision events occur at integer times.
        w *= 2;
        bool node1_boundary = is_boundary_node(e.node1);
        bool node2_boundary = is_boundary_node(e.node2);
        if (node2_boundary && !node1_boundary) {
            boundary_edge_func(e.node1, w, e.observable_indices);
        } else if (node1_boundary && !node2_boundary) {
            boundary_edge_func(e.node2, w, e.observable_indices);
        } else if (!node1_boundary) {
            edge_func(e.node1, e.node2, w, e.observable_indices);
        }
    }
    return normalising_constant * 2;
}

UserGraph vector_checkmatrix_to_user_graph(const std::vector<std::vector<uint8_t>> &vec_H);

UserGraph vector_checkmatrix_to_user_graph(
           const std::vector<std::vector<uint8_t>> &vec_H,
           const std::vector<double> &weights,
           const std::vector<double> &error_probabilities,
           const std::string &merge_strategy,
           bool use_virtual_boundary_node,
           size_t num_repetitions,
           const std::vector<double> &timelike_weights,
           const std::vector<double> &measurement_error_probabilities,
           std::vector<std::vector<uint8_t>> &vec_F);

UserGraph csccheckmatrix_to_user_graph(
           const CSCCheckMatrix &H,
           const std::vector<double> &weights,
           const std::vector<double> &error_probabilities,
           const std::string &merge_strategy,
           bool use_virtual_boundary_node,
           size_t num_repetitions,
           const std::vector<double> &timelike_weights,
           const std::vector<double> &measurement_error_probabilities,
           CSCCheckMatrix &F);

std::vector<uint8_t> decode(UserGraph &graph, const std::vector<uint8_t> &syndrome);

UserGraph detector_error_model_to_user_graph(const stim::DetectorErrorModel& detector_error_model);

}  // namespace pm

#endif  // PYMATCHING2_USER_GRAPH_H
