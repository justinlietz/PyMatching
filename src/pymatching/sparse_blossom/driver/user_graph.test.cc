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
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"

#include <cmath>
#include <gtest/gtest.h>

TEST(UserGraph, GraphFromOnlyVectors) {
    std::vector<std::vector<uint8_t>> vec_H({{1, 1, 0, 0, 0},
                                              {0, 1, 1, 0, 0},
                                              {0, 0, 1, 1, 0},
                                              {0, 0, 0, 1, 1}});

    pm::UserGraph graph = pm::vector_checkmatrix_to_user_graph(vec_H);

    std::vector<uint8_t> syndrome = {0, 1, 0, 0};
    std::vector<uint8_t> decode_output = pm::decode(graph, syndrome);
    /* std::cout << "decoder_output: " << decode_output.size() << "\n"; */
    /* for(size_t i = 0; i < decode_output.size(); ++i){ */
    /*   std::cout << "i: " << i << " val: " << +decode_output[i] << "\n"; */
    /* } */

    std::vector<uint8_t> answer = {1, 1, 0, 0, 0};
    ASSERT_EQ(decode_output, answer);
}


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
    double eprob = -1.0;
    std::vector<double> error_probs(vec_H[0].size(), eprob);
    std::vector<double> meas_error_probs(vec_H[0].size(), eprob);
    double weight = 1.0;
    std::vector<double> weights(vec_H[0].size(), weight);
    std::vector<double> timelike_weights(vec_H[0].size(), weight);
    size_t num_repetitions = 1;
    const std::string merge_strategy("smallest-weight");
    const bool use_virtual_boundary_node = false;
    pm::UserGraph graph = pm::vector_checkmatrix_to_user_graph(vec_H, weights, error_probs,
        merge_strategy, use_virtual_boundary_node, num_repetitions, timelike_weights, meas_error_probs, vec_F);


    std::vector<uint8_t> syndrome = {0, 1, 0, 0};
    std::vector<uint8_t> decode_output = pm::decode(graph, syndrome);
    /* double rweight; */
    /* auto [correction, rweight] = decode(graph, syndrome); */
    /* std::cout << "correc: " << decode_output.size() << "\n"; */
    /* for(size_t i = 0; i < decode_output.size(); ++i){ */
    /*   std::cout << "i: " << i << " val: " << +decode_output[i] << "\n"; */
    /* } */

    std::vector<uint8_t> answer = {1, 1, 0, 0, 0};
    ASSERT_EQ(decode_output, answer);
}

TEST(UserGraph, GraphFromCSCCheckMatrix) {
    std::vector<std::vector<uint8_t>> vec_H({{1, 1, 0, 0, 0},
                                              {0, 1, 1, 0, 0},
                                              {0, 0, 1, 1, 0},
                                              {0, 0, 0, 1, 1}});

    // from here to constructor is all graph optional params
    std::vector<std::vector<uint8_t>> vec_F({{1, 0, 0, 0, 0},
                                              {0, 1, 0, 0, 0},
                                              {0, 0, 1, 0, 0},
                                              {0, 0, 0, 1, 0},
                                              {0, 0, 0, 0, 1}});
    double eprob = -1.0;
    std::vector<double> error_probs(vec_H[0].size(), eprob);
    std::vector<double> meas_error_probs(vec_H[0].size(), eprob);
    double weight = 1.0;
    std::vector<double> weights(vec_H[0].size(), weight);
    std::vector<double> timelike_weights(vec_H[0].size(), weight);
    size_t num_repetitions = 1;
    const std::string merge_strategy("smallest-weight");
    const bool use_virtual_boundary_node = false;
    pm::CSCCheckMatrix csc_H(vec_H);
    pm::CSCCheckMatrix csc_F(vec_F);
    pm::UserGraph graph = pm::csccheckmatrix_to_user_graph(csc_H, weights, error_probs,
        merge_strategy, use_virtual_boundary_node, num_repetitions, timelike_weights, meas_error_probs, csc_F);


    std::vector<uint8_t> syndrome = {0, 1, 0, 0};
    std::vector<uint8_t> decode_output = pm::decode(graph, syndrome);
    std::vector<uint8_t> answer = {1, 1, 0, 0, 0};
    ASSERT_EQ(decode_output, answer);
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
