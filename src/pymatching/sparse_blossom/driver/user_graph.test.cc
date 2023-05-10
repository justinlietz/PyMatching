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

TEST(UserGraph, GraphFromVectors) {
    std::vector<std::vector<uint8_t>> vec_H({{1, 1, 0, 0, 0},
                                              {0, 1, 1, 0, 0},
                                              {0, 0, 1, 1, 0},
                                              {0, 0, 0, 1, 1}});
    std::vector<std::vector<uint8_t>> vec_F({{1, 0, 0, 0, 0},
                                              {0, 1, 0, 0, 0},
                                              {0, 0, 1, 0, 0},
                                              {0, 0, 0, 1, 0},
                                              {0, 0, 0, 0, 1}});
    size_t H_rows = 4;
    size_t H_cols = 5;

    double eprob = -1.0;
    std::vector<double> error_probs(vec_H[0].size(), eprob);
    std::vector<double> meas_error_probs(vec_H[0].size(), eprob);
    double weight = 1.0;
    std::vector<double> weights(vec_H[0].size(), weight);
    std::vector<double> timelike_weights(vec_H[0].size(), weight);

    size_t num_repetitions = 1;
    size_t num_detectors = H_rows * num_repetitions;
    /* int my_val = test_func(2); */
    /* std::cout << "myval: " << my_val << "\n"; */
    /* pm::UserGraph graph(num_detectors, F_rows); */
    const std::string merge_strategy("smallest-weight");
    const bool use_virtual_boundary_node = false;
    pm::UserGraph graph = pm::vector_checkmatrix_to_user_graph(vec_H, weights, error_probs,
        merge_strategy, use_virtual_boundary_node, num_repetitions, timelike_weights, meas_error_probs, vec_F);
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

    std::vector<uint8_t> syndrome = {0, 1, 0, 0};
    // why is this int64 in mwpw_decoding?
    /* std::vector<uint64_t> detection_events = {0, 1, 0, 0, 0}; */
    std::vector<uint64_t> detection_events = {1};
    // detection_events = z.nonzero()[0]
    std::vector<uint8_t> obs_crossed(graph.get_num_observables());
    std::cout << "num obs: " << graph.get_num_observables() << "\n";
    std::cout << "obs: " << obs_crossed.size() << "\n";
    /* pm::total_weight_int weight2 = 0; */
    pm::total_weight_int weight2 = 0;
    pm::decode_detection_events(mwpm, detection_events, obs_crossed.data(), weight2);
    double rescaled_weight = (double)weight / mwpm.flooder.graph.normalising_constant;
    std::cout << "obs: " << obs_crossed.size() << "\n";
    std::cout << "correction: \n";
    for(size_t i = 0; i < obs_crossed.size(); ++i){
      std::cout << "i: " << i << " val: " << +obs_crossed[i] << "\n";
    }
    std::cout << "\n" << "weight: " << rescaled_weight << "\n";

    std::vector<uint8_t> correc = pm::decode(graph, syndrome);
    /* double rweight; */
    /* auto [correc, rweight] = decode(graph, syndrome); */
    std::cout << "correc: " << correc.size() << "\n";
    for(size_t i = 0; i < correc.size(); ++i){
      std::cout << "i: " << i << " val: " << +correc[i] << "\n";
    }

    /* correction, weight = matching_graph.decode(detection_events); */
    std::vector<uint8_t> answer = {1, 1, 0, 0, 0};
    ASSERT_EQ(correc, answer);
}


TEST(UserGraph, GraphFromCSCCheckMatrix) {
    std::vector<std::vector<uint8_t>> vec_H({{1, 1, 0, 0, 0},
                                              {0, 1, 1, 0, 0},
                                              {0, 0, 1, 1, 0},
                                              {0, 0, 0, 1, 1}});
    std::vector<std::vector<uint8_t>> vec_F({{1, 0, 0, 0, 0},
                                              {0, 1, 0, 0, 0},
                                              {0, 0, 1, 0, 0},
                                              {0, 0, 0, 1, 0},
                                              {0, 0, 0, 0, 1}});
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
    double eprob = -1.0;
    std::vector<double> error_probs(vec_H[0].size(), eprob);
    std::vector<double> meas_error_probs(vec_H[0].size(), eprob);
    /* std::cout << "prob size: " << error_probs.size() << "\n"; */
    double weight = 1.0;
    std::vector<double> weights(vec_H[0].size(), weight);
    std::vector<double> timelike_weights(vec_H[0].size(), weight);

    /* auto H = CompressedSparseColumnCheckMatrix(rows_H); */
    /* auto F = CompressedSparseColumnCheckMatrix(rows_F); */
    auto H = pm::CSCCheckMatrix(vec_H);
    auto F = pm::CSCCheckMatrix(vec_F);
    H.print_dense();
    /* return; */

    /* auto H = CSCCheckMatrix(H_data, H_inds, H_ptrs, H_rows, H_cols); */
    /* auto F = CSCCheckMatrix(F_data, F_inds, F_ptrs, F_rows, F_cols); */
    // is is square with size ncolsxncols
    size_t num_repetitions = 1;
    size_t num_detectors = H_rows * num_repetitions;
    /* int my_val = test_func(2); */
    /* std::cout << "myval: " << my_val << "\n"; */
    /* pm::UserGraph graph(num_detectors, F_rows); */
    const std::string merge_strategy("smallest-weight");
    const bool use_virtual_boundary_node = false;
    pm::UserGraph graph = pm::csccheckmatrix_to_user_graph(H, weights, error_probs,
        merge_strategy, use_virtual_boundary_node, num_repetitions, timelike_weights, meas_error_probs, F);
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

    std::vector<uint8_t> syndrome = {0, 1, 0, 0};
    // why is this int64 in mwpw_decoding?
    /* std::vector<uint64_t> detection_events = {0, 1, 0, 0, 0}; */
    std::vector<uint64_t> detection_events = {1};
    // detection_events = z.nonzero()[0]
    std::vector<uint8_t> obs_crossed(graph.get_num_observables());
    std::cout << "num obs: " << graph.get_num_observables() << "\n";
    std::cout << "obs: " << obs_crossed.size() << "\n";
    /* pm::total_weight_int weight2 = 0; */
    pm::total_weight_int weight2 = 0;
    pm::decode_detection_events(mwpm, detection_events, obs_crossed.data(), weight2);
    double rescaled_weight = (double)weight / mwpm.flooder.graph.normalising_constant;
    std::cout << "obs: " << obs_crossed.size() << "\n";
    std::cout << "correction: \n";
    for(size_t i = 0; i < obs_crossed.size(); ++i){
      std::cout << "i: " << i << " val: " << +obs_crossed[i] << "\n";
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
