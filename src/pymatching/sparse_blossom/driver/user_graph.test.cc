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
/* #include "pymatching/sparse_blossom/driver/user_graph.pybind.h" */
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"

#include <cmath>
#include <gtest/gtest.h>

pm::MERGE_STRATEGY merge_strategy_from_string(const std::string &merge_strategy) {
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
};


pm::UserGraph cpp_sparse_column_check_matrix_to_matching_graph(
           const CSCCheckMatrix &H,
           const CSCCheckMatrix &F,
           const std::vector<double> &weights,
           const std::vector<double> &error_probabilities,
           const std::string &merge_strategy,
           bool use_virtual_boundary_node,
           size_t num_repetitions){
           // const std::vector<double> &timelike_weights,
           // const std::vector<double> &measurement_error_probabilities) {
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

            /* auto weights_unchecked = weights.unchecked<1>(); */
            std::vector<double> weights_unchecked(weights);
            // Check weights array size is correct
            if ((size_t)weights_unchecked.size() != H.num_cols)
                throw std::invalid_argument(
                    "The size of the `weights` array (" + std::to_string(weights_unchecked.size()) +
                    ") should match the number of columns in the check matrix (" + std::to_string(H.num_cols) + ")");
            std::vector<double> error_probabilities_unchecked(error_probabilities);
            /* auto error_probabilities_unchecked = error_probabilities.unchecked<1>(); */
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

            auto merge_strategy_enum = merge_strategy_from_string(merge_strategy);

            /* auto H_indptr_unchecked = H.indptr.unchecked<1>(); */
            /* auto H_indices_unchecked = H.indices.unchecked<1>(); */
            /* auto F_indptr_unchecked = F.indptr.unchecked<1>(); */
            /* auto F_indices_unchecked = F.indices.unchecked<1>(); */
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
                /* for (py::ssize_t c = 0; (size_t)c < H.num_cols; c++) { */
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

            /*
            if (num_repetitions > 1) {
                if (timelike_weights.is(py::none()))
                    throw std::invalid_argument("must provide `timelike_weights` for repetitions > 1.");
                if (measurement_error_probabilities.is(py::none()))
                    throw std::invalid_argument("must provide `measurement_error_probabilities` for repetitions > 1.");
                auto t_weights = timelike_weights.unchecked<1>();
                if ((size_t)t_weights.size() != H.num_rows) {
                    throw std::invalid_argument(
                        "timelike_weights has length " + std::to_string(t_weights.size()) +
                        " but its length must equal the number of columns in the check matrix (" +
                        std::to_string(H.num_rows) + ").");
                }
                auto meas_errs = measurement_error_probabilities.unchecked<1>();
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
                            t_weights(row),
                            meas_errs(row),
                            merge_strategy_enum);
                    }
                }
            }
            */

            // Set the boundary if not using a virtual boundary and if a boundary edge was added
            if (!use_virtual_boundary_node && graph.nodes.size() == num_detectors + 1)
                graph.set_boundary({num_detectors});

            return graph;
}

TEST(UserGraph, NewConstructor) {
    std::vector<std::vector<uint8_t>> rows_H({{1, 1, 0, 0, 0},
                                              {0, 1, 1, 0, 0},
                                              {0, 0, 1, 1, 0},
                                              {0, 0, 0, 1, 1}});
    size_t H_rows = 4;
    size_t H_cols = 5;
    std::vector<uint8_t> H_data = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<uint64_t> H_inds = {0, 0, 1, 1, 2, 2, 3, 3};
    std::vector<uint64_t> H_ptrs = {0, 1, 3, 5, 7, 8};
    size_t F_rows = 5;
    size_t F_cols = 5;
    std::vector<uint8_t> F_data = {1, 1, 1, 1, 1,};
    std::vector<uint64_t> F_inds = {0, 1, 2, 3, 4};
    std::vector<uint64_t> F_ptrs = {0, 1, 2, 3, 4, 5};

    /* std::vector<double> weights = {1.0, 2.0, 3.0, 2.0, 1.5}; */
    /* std::vector<double> error_probs = {0.4, 0.3, 0.4, 0.2, 0.1}; */
    double eprob = 0.1;
    std::vector<double> error_probs = {eprob, eprob, eprob, eprob, eprob};
    double weight = log( (1 - eprob)/eprob );
    std::vector<double> weights = {weight, weight, weight, weight, weight};

    auto H = CSCCheckMatrix(H_data, H_inds, H_ptrs, H_rows, H_cols);
    auto F = CSCCheckMatrix(F_data, F_inds, F_ptrs, F_rows, F_cols);
    // is is square with size ncolsxncols
    size_t num_repetitions = 1;
    size_t num_detectors = H_rows * num_repetitions;
    /* int my_val = test_func(2); */
    /* std::cout << "myval: " << my_val << "\n"; */
    /* pm::UserGraph graph(num_detectors, F_rows); */
    const std::string merge_strategy("smallest-weight");
    const bool use_virtual_boundary_node = false;
    pm::UserGraph graph = cpp_sparse_column_check_matrix_to_matching_graph(H, F, weights, error_probs,
        merge_strategy, use_virtual_boundary_node, num_repetitions);
    auto& mwpm = graph.get_mwpm();

    /* auto err_capsule = py::capsule(obs_crossed, [](void *x) { */
    /*     delete reinterpret_cast<std::vector<uint8_t> *>(x); */
    /* }); */
    /* py::array_t<uint8_t> obs_crossed_arr = */
    /*     py::array_t<uint8_t>(obs_crossed->size(), obs_crossed->data(), err_capsule); */
    /* std::pair<py::array_t<std::uint8_t>, double> res = {obs_crossed_arr, rescaled_weight}; */

    /* pm::ExtendedMatchingResult res(mwpm.flooder.graph.num_observables); */
    /* pm::decode_detection_events(mwpm, {2}, res.obs_crossed.data(), res.weight); */
    /* auto& mg = graph.to_matching_graph(weights.size()); */

    std::vector<uint8_t> syndrome = {1, 1, 0, 0};
    // why is this int64 in mwpw_decoding?
    /* std::vector<uint64_t> detection_events = {0, 1, 0, 0, 0}; */
    std::vector<uint64_t> detection_events = {0, 1};
    // detection_events = z.nonzero()[0]
    std::vector<uint8_t> obs_crossed(graph.get_num_observables(), 0);
    /* pm::total_weight_int weight2 = 0; */
    pm::total_weight_int weight2 = 0;
    pm::decode_detection_events(mwpm, detection_events, obs_crossed.data(), weight2);
    double rescaled_weight = (double)weight / mwpm.flooder.graph.normalising_constant;
    std::cout << "correction: \n";
    for(size_t i = 0; i < obs_crossed.size(); ++i){
      std::cout << obs_crossed[i] << " ";
    }
    std::cout << "\n" << "weight: " << rescaled_weight << "\n";

    /* correction, weight = matching_graph.decode(detection_events); */
    ASSERT_EQ(1, 1);
}


TEST(UserGraph, ConstructGraph) {
    pm::UserGraph graph;
    graph.add_or_merge_boundary_edge(0, {2}, 4.1, 0.1);
    graph.add_or_merge_boundary_edge(0, {3}, 1.0, 0.46, pm::INDEPENDENT);
    graph.add_or_merge_edge(0, 1, {0}, 2.5, 0.4);
    graph.add_or_merge_edge(0, 1, {0}, 2.1, 0.45, pm::INDEPENDENT);
    graph.add_or_merge_edge(1, 2, {1}, -3.5, 0.8);
    graph.add_or_merge_edge(2, 3, {3}, 1.8, 0.3);
    graph.add_or_merge_edge(2, 4, {4}, 2.0, 0.25);
    graph.add_or_merge_boundary_edge(2, {4}, 2.2, 0.25);
    graph.set_boundary({3, 4});
    ASSERT_EQ(graph.get_num_observables(), 5);
    ASSERT_EQ(graph.nodes.size(), 5);
    ASSERT_EQ(graph.nodes[0].neighbors[0].edge_it->node2, SIZE_MAX);
    ASSERT_EQ(graph.nodes[0].neighbors[0].edge_it->weight, pm::merge_weights(4.1, 1.0));
    ASSERT_EQ(graph.nodes[0].neighbors[0].edge_it->error_probability, 0.1 * (1 - 0.46) + 0.46 * (1 - 0.1));
    ASSERT_EQ(graph.nodes[0].neighbors[1].edge_it->node2, 1);
    ASSERT_EQ(graph.nodes[0].neighbors[1].edge_it->weight, pm::merge_weights(2.5, 2.1));
    ASSERT_EQ(graph.nodes[0].neighbors[1].edge_it->error_probability, 0.4 * (1 - 0.45) + 0.45 * (1 - 0.4));
    ASSERT_EQ(graph.nodes[1].neighbors[0].edge_it->weight, pm::merge_weights(2.5, 2.1));
    ASSERT_EQ(graph.nodes[1].neighbors[0].edge_it->error_probability, 0.4 * (1 - 0.45) + 0.45 * (1 - 0.4));
    ASSERT_EQ(graph.nodes[1].neighbors[1].edge_it->weight, -3.5);
    ASSERT_EQ(graph.nodes[1].neighbors[0].edge_it->node1, 0);
    ASSERT_EQ(graph.nodes[2].neighbors[0].edge_it->node1, 1);
    ASSERT_EQ(graph.nodes[2].neighbors[1].edge_it->node2, 3);
    ASSERT_EQ(graph.nodes[0].index_of_neighbor(1), 1);
    ASSERT_EQ(graph.nodes[0].index_of_neighbor(SIZE_MAX), 0);
    ASSERT_EQ(graph.nodes[0].index_of_neighbor(3), SIZE_MAX);
    auto& mwpm = graph.get_mwpm();
    auto& g2 = mwpm.flooder.graph;
    ASSERT_EQ(g2.nodes.size(), 5);
    pm::Neighbor n = {nullptr, 4.1, {2}};
    ASSERT_EQ(g2.nodes[0].neighbors[0], nullptr);
    ASSERT_EQ(
        g2.nodes[0].neighbor_weights[0],
        2 * (pm::weight_int)round(pm::merge_weights(4.1, 1.0) * mwpm.flooder.graph.normalising_constant / 2));
    ASSERT_EQ(g2.nodes[0].neighbor_observables[0], 1 << 2);
    ASSERT_EQ(g2.nodes[0].neighbors[1], &g2.nodes[1]);
    ASSERT_EQ(
        g2.nodes[0].neighbor_weights[1],
        2 * (pm::weight_int)round(pm::merge_weights(2.5, 2.1) * mwpm.flooder.graph.normalising_constant / 2));
    ASSERT_EQ(g2.nodes[0].neighbor_observables[1], 1 << 0);
    ASSERT_EQ(
        g2.nodes[1].neighbor_weights[1], 2 * (pm::weight_int)round(3.5 * mwpm.flooder.graph.normalising_constant / 2));
    ASSERT_EQ(g2.nodes[2].neighbors[0], nullptr);
    ASSERT_EQ(
        g2.nodes[2].neighbor_weights[0], 2 * (pm::weight_int)round(1.8 * mwpm.flooder.graph.normalising_constant / 2));
    ASSERT_EQ(g2.nodes[2].neighbor_observables[0], 1 << 3);
    ASSERT_EQ(g2.nodes[2].neighbors[1], &g2.nodes[1]);
    ASSERT_EQ(
        g2.nodes[2].neighbor_weights[1], 2 * (pm::weight_int)round(3.5 * mwpm.flooder.graph.normalising_constant / 2));
    ASSERT_EQ(g2.nodes[2].neighbor_observables[1], 1 << 1);
    ASSERT_EQ(g2.nodes[2].neighbors.size(), 2);
    ASSERT_EQ(g2.nodes[3].neighbors.size(), 0);
    ASSERT_EQ(
        mwpm.flooder.negative_weight_sum,
        -2 * (pm::total_weight_int)round(3.5 * mwpm.flooder.graph.normalising_constant / 2));
    std::set<size_t> dets_exp = {1, 2};
    ASSERT_EQ(mwpm.flooder.graph.negative_weight_detection_events_set.size(), 2);
    std::set<size_t> obs_exp = {1};
    ASSERT_EQ(mwpm.flooder.graph.negative_weight_observables_set, obs_exp);
    ASSERT_EQ(mwpm.flooder.negative_weight_obs_mask, 1 << 1);
    std::vector<uint64_t> dets_exp_vec = {1, 2};
    ASSERT_EQ(mwpm.flooder.negative_weight_detection_events, dets_exp_vec);
    std::vector<size_t> obs_exp_vec = {1};
    ASSERT_EQ(mwpm.flooder.negative_weight_observables, obs_exp_vec);
}

TEST(UserGraph, AddNoise) {
    pm::UserGraph graph;
    graph.add_or_merge_boundary_edge(0, {0}, 1, 1);
    graph.add_or_merge_edge(0, 1, {1}, 1, 0);
    graph.add_or_merge_edge(1, 2, {2}, 1, 0);
    graph.add_or_merge_edge(2, 3, {3}, 1, 1);
    graph.add_or_merge_edge(3, 4, {4}, 1, 0);
    graph.add_or_merge_edge(4, 5, {5}, 1, 0);
    graph.add_or_merge_edge(5, 6, {6}, 1, 1);
    graph.add_or_merge_edge(6, 7, {7}, 1, 1);
    graph.set_boundary({7});
    std::vector<uint8_t> observables(graph.get_num_observables());
    std::vector<uint8_t> syndrome(graph.get_num_nodes());
    graph.add_noise(observables.data(), syndrome.data());
    std::vector<uint8_t> expected_observables = {1, 0, 0, 1, 0, 0, 1, 1};
    ASSERT_EQ(observables, expected_observables);
    std::vector<uint8_t> expected_syndrome = {1, 0, 1, 1, 0, 1, 0, 0};
    ASSERT_EQ(syndrome, expected_syndrome);
}

TEST(UserGraph, NodesAlongShortestPath) {
    pm::UserGraph graph;
    graph.add_or_merge_boundary_edge(0, {0}, 1, -1);
    graph.add_or_merge_edge(0, 1, {1}, 1, -1);
    graph.add_or_merge_edge(1, 2, {2}, 1, -1);
    graph.add_or_merge_edge(2, 3, {3}, 1, -1);
    graph.add_or_merge_edge(3, 4, {4}, 1, -1);
    graph.add_or_merge_edge(4, 5, {5}, 1, -1);
    graph.set_boundary({5});

    {
        std::vector<size_t> nodes;
        graph.get_nodes_on_shortest_path_from_source(4, 0, nodes);
        std::vector<size_t> nodes_expected = {4, 3, 2, 1, 0};
        ASSERT_EQ(nodes, nodes_expected);
    }

    {
        std::vector<size_t> nodes;
        graph.get_nodes_on_shortest_path_from_source(1, SIZE_MAX, nodes);
        std::vector<size_t> nodes_expected = {1, 0};
        ASSERT_EQ(nodes, nodes_expected);
    }

    {
        std::vector<size_t> nodes;
        graph.get_nodes_on_shortest_path_from_source(SIZE_MAX, 3, nodes);
        std::vector<size_t> nodes_expected = {4, 3};
        ASSERT_EQ(nodes, nodes_expected);
    }

    {
        std::vector<size_t> nodes;
        graph.get_nodes_on_shortest_path_from_source(5, 3, nodes);
        std::vector<size_t> nodes_expected = {4, 3};
        ASSERT_EQ(nodes, nodes_expected);
    }
}

TEST(UserGraph, DecodeUserGraphDetectionEventOnBoundaryNode) {
    {
        pm::UserGraph graph;
        graph.add_or_merge_edge(0, 1, {0}, 1.0, -1);
        graph.add_or_merge_edge(1, 2, {1}, 1.0, -1);
        graph.set_boundary({2});
        auto& mwpm = graph.get_mwpm();
        pm::ExtendedMatchingResult res(mwpm.flooder.graph.num_observables);
        pm::decode_detection_events(mwpm, {2}, res.obs_crossed.data(), res.weight);
    }

    {
        pm::UserGraph graph;
        graph.add_or_merge_edge(0, 1, {0}, -1.0, -1);
        graph.add_or_merge_edge(1, 2, {1}, 1.0, -1);
        graph.set_boundary({2});
        auto& mwpm = graph.get_mwpm();
        pm::ExtendedMatchingResult res(mwpm.flooder.graph.num_observables);
        pm::decode_detection_events(mwpm, {2}, res.obs_crossed.data(), res.weight);
    }
}
