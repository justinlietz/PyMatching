#include "gtest/gtest.h"

#include "pymatching/fill_match/search/search_flooder.h"

TEST(SearchFlooder, RepCodeDetectorSearch) {
    size_t num_nodes = 40;
    auto flooder = pm::SearchFlooder(pm::SearchGraph(num_nodes));
    auto& g = flooder.graph;
    g.add_boundary_edge(0, 2, {0});
    for (size_t i = 0; i < num_nodes - 1; i++)
        g.add_edge(i, i+1, 2, {i+1});

    auto collision_edge = flooder.run_until_collision(&g.nodes[1], &g.nodes[20]);
    ASSERT_EQ(collision_edge.collision_node, &g.nodes[10]);
    ASSERT_EQ(collision_edge.neighbor_index, 1);
    ASSERT_EQ(flooder.target_type, pm::DETECTOR_NODE);
    std::vector<uint8_t> observables(num_nodes, 0);
    pm::cumulative_time_int weight = 0;
    flooder.trace_back_path_from_collision_edge(collision_edge, observables.begin(), weight);
    std::vector<uint8_t> expected_obs(num_nodes, 0);
    for (size_t i = 1; i < 20; i++)
        expected_obs[i+1] ^= 1;
    ASSERT_EQ(observables, expected_obs);
    ASSERT_EQ(weight,38);
    flooder.reset_graph();
    for (auto& n : g.nodes) {
        ASSERT_EQ(n.index_of_predecessor, SIZE_MAX);
        ASSERT_EQ(n.reached_from_source, nullptr);
        ASSERT_EQ(n.node_event_tracker.desired_time, pm::cyclic_time_int{0});
        ASSERT_EQ(n.node_event_tracker.queued_time, pm::cyclic_time_int{0});
        ASSERT_EQ(n.node_event_tracker.has_desired_time, false);
        ASSERT_EQ(n.node_event_tracker.has_queued_time, false);
    }
    ASSERT_EQ(flooder.reached_nodes.size(), 0);
}

TEST(SearchFlooder, RepCodeBoundarySearch) {
    size_t num_nodes = 20;
    auto flooder = pm::SearchFlooder(pm::SearchGraph(num_nodes));
    auto& g = flooder.graph;
    g.add_boundary_edge(0, 2, {0});
    for (size_t i = 0; i < num_nodes - 1; i++)
        g.add_edge(i, i+1, 2, {i+1});

    auto collision_edge = flooder.run_until_collision(&g.nodes[6], nullptr);
    ASSERT_EQ(collision_edge.collision_node, &g.nodes[0]);
    ASSERT_EQ(collision_edge.neighbor_index, 0);
    ASSERT_EQ(flooder.target_type, pm::BOUNDARY);
    std::vector<uint8_t> observables(num_nodes, 0);
    pm::cumulative_time_int weight = 0;
    flooder.trace_back_path_from_collision_edge(collision_edge, observables.begin(), weight);
    std::vector<uint8_t> expected_obs(num_nodes, 0);
    for (size_t i = 0; i < 7; i++)
        expected_obs[i] ^= 1;
    ASSERT_EQ(observables, expected_obs);
    ASSERT_EQ(weight, 14);
}
