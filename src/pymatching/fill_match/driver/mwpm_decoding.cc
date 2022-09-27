#include "pymatching/fill_match/driver/mwpm_decoding.h"

void pm::fill_bit_vector_from_obs_mask(
        pm::obs_int obs_mask,
        uint8_t *obs_begin_ptr,
        size_t num_observables
        ){
    auto max_obs = sizeof(pm::obs_int) * 8;
    if (num_observables > max_obs)
        throw std::invalid_argument("Too many observables");
    for (size_t i = 0; i < num_observables; i++)
        *(obs_begin_ptr + i) ^= (obs_mask & (1 << i)) >> i;
}

pm::obs_int pm::bit_vector_to_obs_mask(const std::vector<uint8_t>& bit_vector){
    auto num_observables = bit_vector.size();
    auto max_obs = sizeof(pm::obs_int) * 8;
    if (num_observables > max_obs)
        throw std::invalid_argument("Too many observables");
    pm::obs_int obs_mask = 0;
    for (size_t i = 0; i < num_observables; i++)
        obs_mask ^= bit_vector[i] << i;
    return obs_mask;
}

pm::Mwpm pm::detector_error_model_to_mwpm(
    const stim::DetectorErrorModel& detector_error_model, pm::weight_int num_distinct_weights) {
    auto probability_graph = pm::detector_error_model_to_probability_graph(detector_error_model);
    if (probability_graph.num_observables > sizeof(pm::obs_int) * 8) {
        return pm::Mwpm(
                pm::GraphFlooder(probability_graph.to_matching_graph(num_distinct_weights)),
                pm::SearchFlooder(probability_graph.to_search_graph(num_distinct_weights))
                        );
    } else {
        return pm::Mwpm(pm::GraphFlooder(
                probability_graph.to_matching_graph(num_distinct_weights)
        ));
    }
}


void process_timeline_until_completion(pm::Mwpm& mwpm, const std::vector<uint64_t>& detection_events) {
    if (!mwpm.flooder.queue.empty()) {
        throw std::invalid_argument("!mwpm.flooder.queue.empty()");
    }
    mwpm.flooder.queue.cur_time = 0;

    // Add detection events
    for (auto& detection : detection_events) {
        if (detection >= mwpm.flooder.graph.nodes.size())
            throw std::invalid_argument("Detection event index too large");
        mwpm.create_detection_event(&mwpm.flooder.graph.nodes[detection]);
    }

    while (true) {
        auto event = mwpm.flooder.run_until_next_mwpm_notification();
        if (event.event_type == pm::NO_EVENT)
            break;
        mwpm.process_event(event);
    }
}


pm::MatchingResult shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(
        pm::Mwpm& mwpm, const std::vector<uint64_t>& detection_events) {
    pm::MatchingResult res;
    for (auto& i : detection_events) {
        if (mwpm.flooder.graph.nodes[i].region_that_arrived)
            res += mwpm.shatter_blossom_and_extract_matches(mwpm.flooder.graph.nodes[i].region_that_arrived_top);
    }
    return res;
}


pm::MatchingResult pm::decode_detection_events_for_up_to_64_observables(pm::Mwpm& mwpm, const std::vector<uint64_t>& detection_events) {
    process_timeline_until_completion(mwpm, detection_events);
    return shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(mwpm, detection_events);
}

void pm::decode_detection_events(
        pm::Mwpm& mwpm, const std::vector<uint64_t>& detection_events,
        uint8_t *obs_begin_ptr,
        pm::cumulative_time_int& weight) {
    size_t num_observables = mwpm.flooder.graph.num_observables;
    process_timeline_until_completion(mwpm, detection_events);
    if (num_observables > sizeof(pm::obs_int) * 8) {
        mwpm.flooder.match_edges.clear();
        for (auto& i : detection_events) {
            if (mwpm.flooder.graph.nodes[i].region_that_arrived)
                mwpm.shatter_blossom_and_extract_match_edges(
                        mwpm.flooder.graph.nodes[i].region_that_arrived_top,
                        mwpm.flooder.match_edges
                        );
        }
        mwpm.extract_paths_from_match_edges(mwpm.flooder.match_edges, obs_begin_ptr,weight);
    } else {
        pm::MatchingResult bit_packed_res = shatter_blossoms_for_all_detection_events_and_extract_obs_mask_and_weight(
                mwpm, detection_events
                );
        fill_bit_vector_from_obs_mask(bit_packed_res.obs_mask, obs_begin_ptr,
                                      num_observables);
        weight = bit_packed_res.weight;
    }
}
