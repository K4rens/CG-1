#pragma once
#include "graphics_internal.hpp"
#include <glm/glm.hpp>

namespace application {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

bool initialize();
void shutdown();
void update(double time);
void render(const graphics::internal::FrameData& fd);

}