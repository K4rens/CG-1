#pragma once

#include "graphics_internal.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace application {

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

extern std::vector<Vertex> sphereVertices;
extern std::vector<uint32_t> sphereIndices;

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application