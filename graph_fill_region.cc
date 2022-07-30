#include "graph_fill_region.h"
#include "varying.h"
#include "graph.h"

pm::GraphFillRegion::GraphFillRegion()
    : blossom_parent(nullptr), alt_tree_node(nullptr), radius((0 << 2) + 1) {}


const pm::GraphFillRegion* pm::GraphFillRegion::top_region() const {
    auto current = this;
    while (current->blossom_parent) {
        current = current->blossom_parent;
    }
    return current;
}


bool pm::GraphFillRegion::tree_equal(const pm::GraphFillRegion& other) const {
    if (
            alt_tree_node != other.alt_tree_node || radius != other.radius ||
            blossom_parent != other.blossom_parent ||
            blossom_children.size() != other.blossom_children.size() ||
            shell_area != other.shell_area
    ) {
        return false;
    }
    if (blossom_children.empty())
        return true;
    for (size_t i = 0; i < blossom_children.size(); i++) {
        if (blossom_children[i].edge != other.blossom_children[i].edge)
            return false;
        if (!blossom_children[i].region->tree_equal(*other.blossom_children[i].region))
            return false;
    }
    return true;
}


bool pm::GraphFillRegion::operator==(const pm::GraphFillRegion &rhs) const {
    return tree_equal(rhs);
}

bool pm::GraphFillRegion::operator!=(const pm::GraphFillRegion &rhs) const {
    return !(rhs == *this);
}

void pm::GraphFillRegion::add_match(pm::GraphFillRegion *region, const pm::CompressedEdge &edge) {
    match = Match(region, edge);
    region->match = Match(this,edge.reversed());
}

pm::Match::Match(pm::GraphFillRegion *region, pm::CompressedEdge edge) : region(region), edge(edge) {}

pm::Match::Match() : region(nullptr), edge(CompressedEdge()) {}
