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

#include "pymatching/sparse_blossom/driver/user_graph.h"

// this include means this should probably be in a separate file
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"

pm::UserNode::UserNode() : is_boundary(false) {
}

size_t pm::UserNode::index_of_neighbor(size_t node) const {
    auto it = std::find_if(neighbors.begin(), neighbors.end(), [&](const UserNeighbor& neighbor) {
        if (neighbor.pos == 0) {
            return neighbor.edge_it->node1 == node;
        } else if (neighbor.pos == 1) {
            return neighbor.edge_it->node2 == node;
        } else {
            throw std::runtime_error("`neighbor.pos` should be 0 or 1, but got: " + std::to_string(neighbor.pos));
        }
    });
    if (it == neighbors.end())
        return SIZE_MAX;

    return it - neighbors.begin();
}

bool is_valid_probability(double p) {
    return (p >= 0 && p <= 1);
}

void pm::UserGraph::merge_edge_or_boundary_edge(
    size_t node,
    size_t neighbor_index,
    const std::vector<size_t>& parallel_observables,
    double parallel_weight,
    double parallel_error_probability,
    pm::MERGE_STRATEGY merge_strategy) {
    auto& neighbor = nodes[node].neighbors[neighbor_index];
    if (merge_strategy == DISALLOW) {
        throw std::invalid_argument(
            "Edge (" + std::to_string(neighbor.edge_it->node1) + ", " + std::to_string(neighbor.edge_it->node2) +
            ") already exists in the graph. "
            "Parallel edges not permitted with the provided `disallow` `merge_strategy`. Please provide a "
            "different `merge_strategy`.");
    } else if (
        merge_strategy == KEEP_ORIGINAL ||
        (merge_strategy == SMALLEST_WEIGHT && parallel_weight >= neighbor.edge_it->weight)) {
        return;
    } else {
        double new_weight, new_error_probability;
        bool use_new_observables;
        if (merge_strategy == REPLACE || merge_strategy == SMALLEST_WEIGHT) {
            new_weight = parallel_weight;
            new_error_probability = parallel_error_probability;
            use_new_observables = true;
        } else if (merge_strategy == INDEPENDENT) {
            new_weight = pm::merge_weights(parallel_weight, neighbor.edge_it->weight);
            new_error_probability = -1;
            if (is_valid_probability(neighbor.edge_it->error_probability) &&
                is_valid_probability(parallel_error_probability))
                new_error_probability = parallel_error_probability * (1 - neighbor.edge_it->error_probability) +
                                        neighbor.edge_it->error_probability * (1 - parallel_error_probability);
            // We do not need to update the observables. If they do not match up, then the code has distance 2.
            use_new_observables = false;
        } else {
            throw std::invalid_argument("Merge strategy not recognised.");
        }
        // Update the existing edge weight and probability in the adjacency list of `node`
        neighbor.edge_it->weight = new_weight;
        neighbor.edge_it->error_probability = new_error_probability;
        if (use_new_observables)
            neighbor.edge_it->observable_indices = parallel_observables;

        _mwpm_needs_updating = true;
        if (new_error_probability < 0 || new_error_probability > 1)
            _all_edges_have_error_probabilities = false;
    }
}

void pm::UserGraph::add_or_merge_edge(
    size_t node1,
    size_t node2,
    const std::vector<size_t>& observables,
    double weight,
    double error_probability,
    MERGE_STRATEGY merge_strategy) {
    auto max_id = std::max(node1, node2);
    if (max_id + 1 > nodes.size())
        nodes.resize(max_id + 1);

    size_t idx = nodes[node1].index_of_neighbor(node2);

    if (idx == SIZE_MAX) {
        pm::UserEdge edge = {node1, node2, observables, weight, error_probability};
        edges.push_back(edge);
        nodes[node1].neighbors.push_back({std::prev(edges.end()), 1});
        if (node1 != node2)
            nodes[node2].neighbors.push_back({std::prev(edges.end()), 0});

        for (auto& obs : observables) {
            if (obs + 1 > _num_observables)
                _num_observables = obs + 1;
        }
        _mwpm_needs_updating = true;
        if (error_probability < 0 || error_probability > 1)
            _all_edges_have_error_probabilities = false;
    } else {
        merge_edge_or_boundary_edge(node1, idx, observables, weight, error_probability, merge_strategy);
    }
}

void pm::UserGraph::add_or_merge_boundary_edge(
    size_t node,
    const std::vector<size_t>& observables,
    double weight,
    double error_probability,
    MERGE_STRATEGY merge_strategy) {
    if (node + 1 > nodes.size())
        nodes.resize(node + 1);

    size_t idx = nodes[node].index_of_neighbor(SIZE_MAX);

    if (idx == SIZE_MAX) {
        pm::UserEdge edge = {node, SIZE_MAX, observables, weight, error_probability};
        edges.push_back(edge);
        nodes[node].neighbors.push_back({std::prev(edges.end()), 1});

        for (auto& obs : observables) {
            if (obs + 1 > _num_observables)
                _num_observables = obs + 1;
        }
        _mwpm_needs_updating = true;
        if (error_probability < 0 || error_probability > 1)
            _all_edges_have_error_probabilities = false;
    } else {
        merge_edge_or_boundary_edge(node, idx, observables, weight, error_probability, merge_strategy);
    }
}

pm::UserGraph::UserGraph()
    : _num_observables(0), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true) {
}

pm::UserGraph::UserGraph(size_t num_nodes)
    : _num_observables(0), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true) {
    nodes.resize(num_nodes);
}

pm::UserGraph::UserGraph(size_t num_nodes, size_t num_observables)
    : _num_observables(num_observables), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true) {
    nodes.resize(num_nodes);
}

void pm::UserGraph::set_boundary(const std::set<size_t>& boundary) {
    for (auto& n : boundary_nodes)
        nodes[n].is_boundary = false;
    boundary_nodes = boundary;
    for (auto& n : boundary_nodes) {
        if (n >= nodes.size())
            nodes.resize(n + 1);
        nodes[n].is_boundary = true;
    }
    _mwpm_needs_updating = true;
}

std::set<size_t> pm::UserGraph::get_boundary() {
    return boundary_nodes;
}

size_t pm::UserGraph::get_num_observables() {
    return _num_observables;
}

size_t pm::UserGraph::get_num_nodes() {
    return nodes.size();
}

size_t pm::UserGraph::get_num_detectors() {
    return get_num_nodes() - boundary_nodes.size();
}

bool pm::UserGraph::is_boundary_node(size_t node_id) {
    return (node_id == SIZE_MAX) || nodes[node_id].is_boundary;
}

void pm::UserGraph::update_mwpm() {
    _mwpm = to_mwpm(pm::NUM_DISTINCT_WEIGHTS, false);
    _mwpm_needs_updating = false;
}

pm::Mwpm& pm::UserGraph::get_mwpm() {
    if (_mwpm_needs_updating)
        update_mwpm();
    return _mwpm;
}

void pm::UserGraph::add_noise(uint8_t* error_arr, uint8_t* syndrome_arr) const {
    if (!_all_edges_have_error_probabilities)
        return;

    for (auto& e : edges) {
        auto p = e.error_probability;
        if (rand_float(0.0, 1.0) < p) {
            // Flip the observables
            for (auto& obs : e.observable_indices) {
                *(error_arr + obs) ^= 1;
            }
            // Flip the syndrome bits
            *(syndrome_arr + e.node1) ^= 1;
            if (e.node2 != SIZE_MAX)
                *(syndrome_arr + e.node2) ^= 1;
        }
    }

    for (auto& b : boundary_nodes)
        *(syndrome_arr + b) = 0;
}

size_t pm::UserGraph::get_num_edges() {
    return edges.size();
}

bool pm::UserGraph::all_edges_have_error_probabilities() {
    return _all_edges_have_error_probabilities;
}

double pm::UserGraph::max_abs_weight() {
    double max_abs_weight = 0;
    for (auto& e : edges) {
        if (std::abs(e.weight) > max_abs_weight) {
            max_abs_weight = std::abs(e.weight);
        }
    }
    return max_abs_weight;
}

pm::MatchingGraph pm::UserGraph::to_matching_graph(pm::weight_int num_distinct_weights) {
    pm::MatchingGraph matching_graph(nodes.size(), _num_observables);
    double normalising_constant = iter_discretized_edges(
        num_distinct_weights,
        [&](size_t u, size_t v, pm::signed_weight_int weight, const std::vector<size_t>& observables) {
            matching_graph.add_edge(u, v, weight, observables);
        },
        [&](size_t u, pm::signed_weight_int weight, const std::vector<size_t>& observables) {
            // Only add the boundary edge if it already isn't present. Ideally parallel edges should already have been
            // merged, however we are implicitly merging all boundary nodes in this step, which could give rise to new
            // parallel edges.
            if (matching_graph.nodes[u].neighbors.empty() || matching_graph.nodes[u].neighbors[0])
                matching_graph.add_boundary_edge(u, weight, observables);
        });
    matching_graph.normalising_constant = normalising_constant;
    if (boundary_nodes.size() > 0) {
        matching_graph.is_user_graph_boundary_node.clear();
        matching_graph.is_user_graph_boundary_node.resize(nodes.size(), false);
        for (auto& i : boundary_nodes)
            matching_graph.is_user_graph_boundary_node[i] = true;
    }
    return matching_graph;
}

pm::SearchGraph pm::UserGraph::to_search_graph(pm::weight_int num_distinct_weights) {
    /// Identical to to_matching_graph but for constructing a pm::SearchGraph
    pm::SearchGraph search_graph(nodes.size());
    iter_discretized_edges(
        num_distinct_weights,
        [&](size_t u, size_t v, pm::signed_weight_int weight, const std::vector<size_t>& observables) {
            search_graph.add_edge(u, v, weight, observables);
        },
        [&](size_t u, pm::signed_weight_int weight, const std::vector<size_t>& observables) {
            // Only add the boundary edge if it already isn't present. Ideally parallel edges should already have been
            // merged, however we are implicitly merging all boundary nodes in this step, which could give rise to new
            // parallel edges.
            if (search_graph.nodes[u].neighbors.empty() || search_graph.nodes[u].neighbors[0])
                search_graph.add_boundary_edge(u, weight, observables);
        });
    return search_graph;
}

pm::Mwpm pm::UserGraph::to_mwpm(pm::weight_int num_distinct_weights, bool ensure_search_graph_included) {
    if (_num_observables > sizeof(pm::obs_int) * 8 || ensure_search_graph_included) {
        auto mwpm = pm::Mwpm(
            pm::GraphFlooder(to_matching_graph(num_distinct_weights)),
            pm::SearchFlooder(to_search_graph(num_distinct_weights)));
        mwpm.flooder.sync_negative_weight_observables_and_detection_events();
        return mwpm;
    } else {
        auto mwpm = pm::Mwpm(pm::GraphFlooder(to_matching_graph(num_distinct_weights)));
        mwpm.flooder.sync_negative_weight_observables_and_detection_events();
        return mwpm;
    }
}

pm::Mwpm& pm::UserGraph::get_mwpm_with_search_graph() {
    if (!_mwpm_needs_updating && _mwpm.flooder.graph.nodes.size() == _mwpm.search_flooder.graph.nodes.size()) {
        return _mwpm;
    } else {
        _mwpm = to_mwpm(pm::NUM_DISTINCT_WEIGHTS, true);
        _mwpm_needs_updating = false;
        return _mwpm;
    }
}

void pm::UserGraph::handle_dem_instruction(
    double p, const std::vector<size_t>& detectors, const std::vector<size_t>& observables) {
    if (detectors.size() == 2) {
        add_or_merge_edge(detectors[0], detectors[1], observables, std::log((1 - p) / p), p, INDEPENDENT);
    } else if (detectors.size() == 1) {
        add_or_merge_boundary_edge(detectors[0], observables, std::log((1 - p) / p), p, INDEPENDENT);
    }
}

void pm::UserGraph::get_nodes_on_shortest_path_from_source(size_t src, size_t dst, std::vector<size_t>& out_nodes) {
    auto& mwpm = get_mwpm_with_search_graph();
    bool src_is_boundary = is_boundary_node(src);
    bool dst_is_boundary = is_boundary_node(dst);
    if (src != SIZE_MAX && src >= nodes.size())
        throw std::invalid_argument("node " + std::to_string(src) + " is not in the graph");
    if (dst != SIZE_MAX && dst >= nodes.size())
        throw std::invalid_argument("node " + std::to_string(dst) + " is not in the graph");
    if (!src_is_boundary) {
        size_t dst_tmp = dst_is_boundary ? SIZE_MAX : dst;
        mwpm.search_flooder.iter_edges_on_shortest_path_from_source(src, dst_tmp, [&](const pm::SearchGraphEdge edge) {
            out_nodes.push_back(edge.detector_node - &mwpm.search_flooder.graph.nodes[0]);
        });
        if (!dst_is_boundary)
            out_nodes.push_back(dst);
    } else if (!dst_is_boundary) {
        std::vector<size_t> temp_out_nodes;
        get_nodes_on_shortest_path_from_source(dst, src, temp_out_nodes);
        for (size_t i = 0; i < temp_out_nodes.size(); i++) {
            out_nodes.push_back(temp_out_nodes[temp_out_nodes.size() - 1 - i]);
        }
    } else {
        throw std::invalid_argument("Both the source and destination vertices provided are boundary nodes");
    }
}

bool pm::UserGraph::has_edge(size_t node1, size_t node2) {
    if (node1 >= nodes.size())
        return false;
    return nodes[node1].index_of_neighbor(node2) != SIZE_MAX;
}

bool pm::UserGraph::has_boundary_edge(size_t node) {
    if (node >= nodes.size())
        return false;
    return nodes[node].index_of_neighbor(SIZE_MAX) != SIZE_MAX;
}

void pm::UserGraph::set_min_num_observables(size_t num_observables) {
    if (num_observables > _num_observables)
        _num_observables = num_observables;
}

double pm::UserGraph::get_edge_weight_normalising_constant(size_t max_num_distinct_weights) {
    double max_abs_weight = 0;
    bool all_integral_weight = true;
    for (auto& e : edges) {
        if (std::abs(e.weight) > max_abs_weight)
            max_abs_weight = std::abs(e.weight);

        if (round(e.weight) != e.weight)
            all_integral_weight = false;
    }

    if (max_abs_weight > pm::MAX_USER_EDGE_WEIGHT)
        throw std::invalid_argument(
            "maximum absolute edge weight of " + std::to_string(pm::MAX_USER_EDGE_WEIGHT) + " exceeded.");

    if (all_integral_weight) {
        return 1.0;
    } else {
        pm::weight_int max_half_edge_weight = max_num_distinct_weights - 1;
        return (double)max_half_edge_weight / max_abs_weight;
    }
}

pm::UserGraph pm::detector_error_model_to_user_graph(const stim::DetectorErrorModel& detector_error_model) {
    pm::UserGraph user_graph(detector_error_model.count_detectors(), detector_error_model.count_observables());
    pm::iter_detector_error_model_edges(
        detector_error_model, [&](double p, const std::vector<size_t>& detectors, std::vector<size_t>& observables) {
            user_graph.handle_dem_instruction(p, detectors, observables);
        });
    return user_graph;
}

// This is copied from user_graph.pybind.cc, figure out where is should go
pm::MERGE_STRATEGY merge_strategy_from_string2(const std::string &merge_strategy) {
    static std::unordered_map<std::string, pm::MERGE_STRATEGY> const table = {
        {"disallow", pm::DISALLOW},
        {"independent", pm::INDEPENDENT},
        {"smallest-weight", pm::SMALLEST_WEIGHT},
        {"keep-original", pm::KEEP_ORIGINAL},
        {"replace", pm::REPLACE}};
    auto it = table.find(merge_strategy);
    if (it != table.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Merge strategy \"" + merge_strategy + "\" not recognised.");
    }
}

pm::UserGraph pm::vector_checkmatrix_to_user_graph(
           const std::vector<std::vector<uint8_t>> &vec_H,
           const std::vector<double> &weights,
           const std::vector<double> &error_probabilities,
           const std::string &merge_strategy,
           bool use_virtual_boundary_node,
           size_t num_repetitions,
           const std::vector<double> &timelike_weights,
           const std::vector<double> &measurement_error_probabilities,
           std::vector<std::vector<uint8_t>> &vec_F){
    auto H = pm::CSCCheckMatrix(vec_H);
    auto F = pm::CSCCheckMatrix(vec_F);
    pm::UserGraph graph = pm::csccheckmatrix_to_user_graph(
        H, weights, error_probabilities, merge_strategy,
        use_virtual_boundary_node, num_repetitions,
        timelike_weights, measurement_error_probabilities, F);
    return graph;
}

pm::UserGraph pm::csccheckmatrix_to_user_graph(
           const pm::CSCCheckMatrix &H,
           const std::vector<double> &weights,
           const std::vector<double> &error_probabilities,
           const std::string &merge_strategy,
           bool use_virtual_boundary_node,
           size_t num_repetitions,
           const std::vector<double> &timelike_weights,
           const std::vector<double> &measurement_error_probabilities,
           pm::CSCCheckMatrix &F){
           // py::object &faults_matrix) {
           // auto H = CompressedSparseColumnCheckMatrix(check_matrix);
           // auto H = cppCompressedSparseColumnCheckMatrix(dat, inds, ptrs, nrows, ncols);
           // auto F = cppCompressedSparseColumnCheckMatrix(dat, inds, ptrs, nrows, ncols);

           // if (faults_matrix.is(py::none())) {
           //     faults_matrix =
           //         py::module_::import("scipy.sparse")
           //             .attr("eye")(
           //                 H.num_cols, "dtype"_a = py::module_::import("numpy").attr("uint8"), "format"_a = "csc");
           // }
           // auto F = CompressedSparseColumnCheckMatrix(faults_matrix);

            // auto weights_unchecked = weights.unchecked<1>();
            std::vector<double> weights_unchecked(weights);
            // Check weights array size is correct
            if ((size_t)weights_unchecked.size() != H.num_cols)
                throw std::invalid_argument(
                    "The size of the `weights` array (" + std::to_string(weights_unchecked.size()) +
                    ") should match the number of columns in the check matrix (" + std::to_string(H.num_cols) + ")");
            std::vector<double> error_probabilities_unchecked(error_probabilities);
            // auto error_probabilities_unchecked = error_probabilities.unchecked<1>();
            // Check error_probabilities array is correct
            if ((size_t)error_probabilities_unchecked.size() != H.num_cols)
                throw std::invalid_argument(
                    "The size of the `error_probabilities` array (" +
                    std::to_string(error_probabilities_unchecked.size()) +
                    ") should match the number of columns in the check matrix (" + std::to_string(H.num_cols) + ")");

            if (H.num_cols != F.num_cols)
                throw std::invalid_argument(
                    "`faults_matrix` array with shape (" + std::to_string(F.num_rows) + ", " +
                    std::to_string(F.num_cols) +
                    ") must have the same number of columns as the check matrix, which has shape (" +
                    std::to_string(H.num_rows) + ", " + std::to_string(H.num_cols) + ").");

            auto merge_strategy_enum = merge_strategy_from_string2(merge_strategy);

            // auto H_indptr_unchecked = H.indptr.unchecked<1>();
            // auto H_indices_unchecked = H.indices.unchecked<1>();
            // auto F_indptr_unchecked = F.indptr.unchecked<1>();
            // auto F_indices_unchecked = F.indices.unchecked<1>();
            std::vector<uint64_t> H_indptr_unchecked(H.indptr);
            std::vector<uint64_t> H_indices_unchecked(H.indices);
            std::vector<uint64_t> F_indptr_unchecked(F.indptr);
            std::vector<uint64_t> F_indices_unchecked(F.indices);

            // Now construct the graph
            size_t num_detectors = H.num_rows * num_repetitions;
            pm::UserGraph graph(num_detectors, F.num_rows);
            // Each column corresponds to an edge. Iterate over the columns, adding the edges to the graph.
            // Also iterate over the number of repetitions (in case num_repetitions > 1)
            for (size_t rep = 0; rep < num_repetitions; rep++) {
                // for (py::ssize_t c = 0; (size_t)c < H.num_cols; c++) {
                for (size_t c = 0; (size_t)c < H.num_cols; c++) {
                    auto idx_start = H_indptr_unchecked[c];
                    auto idx_end = H_indptr_unchecked[c + 1];
                    auto num_dets = idx_end - idx_start;
                    if (idx_start > H_indices_unchecked.size() - 1 && idx_start != idx_end)
                        throw std::invalid_argument(
                            "`check_matrix.indptr` elements must not exceed size of `check_matrix.indices`");
                    auto f_idx_start = F_indptr_unchecked[c];
                    auto f_idx_end = F_indptr_unchecked[c + 1];
                    std::vector<size_t> obs;
                    obs.reserve(f_idx_end - f_idx_start);
                    for (auto q = f_idx_start; q < f_idx_end; q++)
                        obs.push_back((size_t)F_indices_unchecked[q]);
                    if (num_dets == 2) {
                        graph.add_or_merge_edge(
                            H_indices_unchecked[idx_start] + H.num_rows * rep,
                            H_indices_unchecked[idx_start + 1] + H.num_rows * rep,
                            obs,
                            weights_unchecked[c],
                            error_probabilities_unchecked[c],
                            merge_strategy_enum);
                    } else if (num_dets == 1) {
                        if (use_virtual_boundary_node) {
                            graph.add_or_merge_boundary_edge(
                                H_indices_unchecked[idx_start] + H.num_rows * rep,
                                obs,
                                weights_unchecked[c],
                                error_probabilities_unchecked[c],
                                merge_strategy_enum);
                        } else {
                            graph.add_or_merge_edge(
                                H_indices_unchecked[idx_start] + H.num_rows * rep,
                                num_detectors,
                                obs,
                                weights_unchecked[c],
                                error_probabilities_unchecked[c],
                                merge_strategy_enum);
                        }
                    } else if (num_dets != 0) {
                        throw std::invalid_argument(
                            "`check_matrix` must contain at most two ones per column, but column " + std::to_string(c) +
                            " has " + std::to_string(num_dets) + " ones.");
                    }
                }
            }

            if (num_repetitions > 1) {
                /* if (timelike_weights.is(py::none())) */
                if (timelike_weights.size() < 1)
                    throw std::invalid_argument("must provide `timelike_weights` for repetitions > 1.");
                /* if (measurement_error_probabilities.is(py::none())) */
                if (measurement_error_probabilities.size() < 1)
                    throw std::invalid_argument("must provide `measurement_error_probabilities` for repetitions > 1.");
                /* auto t_weights = timelike_weights.unchecked<1>(); */
                std::vector<double> t_weights(timelike_weights);
                if ((size_t)t_weights.size() != H.num_rows) {
                    throw std::invalid_argument(
                        "timelike_weights has length " + std::to_string(t_weights.size()) +
                        " but its length must equal the number of columns in the check matrix (" +
                        std::to_string(H.num_rows) + ").");
                }
                /* auto meas_errs = measurement_error_probabilities.unchecked<1>(); */
                std::vector<double> meas_errs(measurement_error_probabilities);
                if ((size_t)meas_errs.size() != H.num_rows) {
                    throw std::invalid_argument(
                        "`measurement_error_probabilities` has length " + std::to_string(meas_errs.size()) +
                        " but its length must equal the number of columns in the check matrix (" +
                        std::to_string(H.num_rows) + ").");
                }

                for (size_t rep = 0; rep < num_repetitions - 1; rep++) {
                    for (size_t row = 0; row < H.num_rows; row++) {
                        graph.add_or_merge_edge(
                            row + rep * H.num_rows,
                            row + (rep + 1) * H.num_rows,
                            {},
                            t_weights[row],
                            meas_errs[row],
                            merge_strategy_enum);
                    }
                }
            }

            // Set the boundary if not using a virtual boundary and if a boundary edge was added
            if (!use_virtual_boundary_node && graph.nodes.size() == num_detectors + 1)
                graph.set_boundary({num_detectors});

            return graph;
}

// const graph?
/* std::pair<std::vector<uint8_t>, double> decode(pm::UserGraph &graph, std::vector<uint8_t> syndrome){ */
std::vector<uint8_t> pm::decode(pm::UserGraph &graph, const std::vector<uint8_t> &syndrome){
  // Todo - check syndrome vals are in {0,1}
  std::vector<uint64_t> detection_events;
  for(size_t idx = 0; idx < syndrome.size(); ++idx){
    if( syndrome[idx] == 1 ) detection_events.push_back(idx);
  }
  auto& mwpm = graph.get_mwpm();
  std::vector<uint8_t> correction(graph.get_num_observables());
  pm::total_weight_int weight = 0;
  pm::decode_detection_events(mwpm, detection_events, correction.data(), weight);
  double rescaled_weight = (double)weight / mwpm.flooder.graph.normalising_constant;

  std::pair<std::vector<uint8_t>, double> result(correction, rescaled_weight);
  /* return result; */
  return correction;
}

