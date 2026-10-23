#include "application.hpp"
#include "graphics_internal.hpp"
#include <imgui.h>
#include <cmath>
#include <cstring>
#include <vk_mem_alloc.h>
#include <fstream>
#include <stdexcept>
#include <array>
#include <string>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace application {

glm::vec3 cameraPos   = {0.0f, 1.5f, 6.0f};
float cameraYaw       = -90.0f;
float cameraPitch     = -10.0f;
float cameraFov       = 45.0f;

glm::vec3 dirLightDir       = {0.5f, 1.0f, 1.2f};
glm::vec3 dirLightColor     = {0.5f, 0.1f, 0.3f};
float dirLightIntensity     = 1.0f;

glm::vec3 ambientColor      = {1.0f, 1.0f, 1.0f};

bool enableDirLight = true;
bool enableAmbient  = true;

glm::vec3 objPosition = {0.0f, 0.0f, 0.0f};
glm::vec3 objRotation = {0.0f, 0.0f, 0.0f};
glm::vec3 objScale    = {1.5f, 1.5f, 1.5f};

glm::vec3 matAlbedo    = {0.8f, 0.3f, 0.2f};
glm::vec3 matSpecular  = {1.0f, 1.0f, 1.0f};
float matShininess     = 64.0f;

struct PointLightCPU {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;
};

struct SpotLightCPU {
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 color;
    float intensity;
    float innerAngle;
    float outerAngle;
};

std::vector<PointLightCPU> pointLights;
std::vector<SpotLightCPU> spotLights;

bool showLightMarkers = true;
bool showSpotCones    = true;

struct alignas(16) GPUPrivatePointLight {
    glm::vec4 position;
    glm::vec4 color;
};

struct alignas(16) GPUPrivateSpotLight {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 color;
};

struct alignas(16) SceneUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 viewPos;
    glm::vec4 ambientColor;
    glm::vec4 dirLightDir;
    glm::vec4 dirLightColor;
    int pointLightCount;
    int spotLightCount;
    int pad1;
    int pad2;
};

struct alignas(16) ObjectUBO {
    glm::mat4 model;
    glm::vec4 albedo;
    glm::vec4 specular;
};

struct PushConstantMarker {
    glm::mat4 mvp;
    glm::vec4 color;
};

const uint32_t MAX_LIGHTS_CAPACITY = 64;

std::vector<Vertex> sphereVertices;
std::vector<uint32_t> sphereIndices;
VkBuffer vertexBuffer;
VmaAllocation vertexBufferAllocation;
VkBuffer indexBuffer;
VmaAllocation indexBufferAllocation;

VkBuffer coneLineBuffer;
VmaAllocation coneLineBufferAllocation;
void* coneLineMappedData = nullptr;
uint32_t coneLineVertexCount = 0;

VkBuffer sceneBuffer;
VmaAllocation sceneBufferAllocation;
void* sceneMappedData = nullptr;

VkBuffer objectBuffer;
VmaAllocation objectBufferAllocation;
void* objectMappedData = nullptr;

VkBuffer pointLightSSBO;
VmaAllocation pointLightSSBOAllocation;
void* pointLightSSBOMappedData = nullptr;

VkBuffer spotLightSSBO;
VmaAllocation spotLightSSBOAllocation;
void* spotLightSSBOMappedData = nullptr;

VkDescriptorSetLayout sceneSetLayout;
VkDescriptorSetLayout objectSetLayout;
VkDescriptorPool descriptorPool;

VkDescriptorSet sceneDescriptorSet;
VkDescriptorSet objectDescriptorSet;

VkPipelineLayout litPipelineLayout;
VkPipeline litPipeline;

VkPipelineLayout unlitPipelineLayout;
VkPipeline unlitMeshPipeline;
VkPipeline unlitLinePipeline;

void* createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, void* data = nullptr) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocResultInfo;
    vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, &allocResultInfo);

    if (data != nullptr && allocResultInfo.pMappedData != nullptr) {
        memcpy(allocResultInfo.pMappedData, data, (size_t)size);
    }
    return allocResultInfo.pMappedData;
}

void generateSphere(float radius, int sectorCount, int stackCount) {
    sphereVertices.clear();
    sphereIndices.clear();

    float sectorStep = 2.0f * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;

    for (int i = 0; i <= stackCount; ++i) {
        float stackAngle = M_PI / 2.0f - i * stackStep;
        float y = radius * sinf(stackAngle);
        float xy = radius * cosf(stackAngle);

        for (int j = 0; j <= sectorCount; ++j) {
            float sectorAngle = j * sectorStep;
            float x = xy * cosf(sectorAngle);
            float z = xy * sinf(sectorAngle);

            glm::vec3 pos = {x, y, z};
            glm::vec3 norm = glm::normalize(pos);

            sphereVertices.push_back({pos, norm});
        }
    }

    for (int i = 0; i < stackCount; ++i) {
        int k1 = i * (sectorCount + 1);
        int k2 = k1 + sectorCount + 1;

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            if (i != 0) {
                sphereIndices.push_back(k1);
                sphereIndices.push_back(k1 + 1);
                sphereIndices.push_back(k2);
            }
            if (i != (stackCount - 1)) {
                sphereIndices.push_back(k1 + 1);
                sphereIndices.push_back(k2 + 1);
                sphereIndices.push_back(k2);
            }
        }
    }
}

void updateConeLines() {
    std::vector<glm::vec3> linePoints;
    const float coneLength = 2.0f;
    const int circleSegments = 16;

    for (const auto& spot : spotLights) {
        glm::vec3 apex = spot.position;
        glm::vec3 dir = glm::normalize(spot.direction);
        float radius = coneLength * tanf(glm::radians(spot.outerAngle));
        glm::vec3 baseCenter = apex + dir * coneLength;

        glm::vec3 arbitrary = (fabs(dir.y) < 0.99f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 right = glm::normalize(glm::cross(dir, arbitrary));
        glm::vec3 up = glm::normalize(glm::cross(right, dir));

        linePoints.push_back(apex);
        linePoints.push_back(baseCenter);

        std::vector<glm::vec3> baseRing(circleSegments);
        for (int i = 0; i < circleSegments; ++i) {
            float angle = 2.0f * M_PI * i / circleSegments;
            baseRing[i] = baseCenter + (right * cosf(angle) + up * sinf(angle)) * radius;
        }

        for (int i = 0; i < circleSegments; ++i) {
            linePoints.push_back(baseRing[i]);
            linePoints.push_back(baseRing[(i + 1) % circleSegments]);
        }

        for (int i = 0; i < circleSegments; i += 4) {
            linePoints.push_back(apex);
            linePoints.push_back(baseRing[i]);
        }
    }

    coneLineVertexCount = static_cast<uint32_t>(linePoints.size());
    if (coneLineVertexCount > 0 && coneLineMappedData != nullptr) {
        memcpy(coneLineMappedData, linePoints.data(), sizeof(glm::vec3) * linePoints.size());
    }
}

std::vector<char> readSpvFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) file.open("../" + filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Shader not found: " + filename);
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

VkShaderModule buildShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    vkCreateShaderModule(graphics::internal::context.device, &createInfo, nullptr, &shaderModule);
    return shaderModule;
}

void initPipelinesAndDescriptors() {
    auto& context = graphics::internal::context;

    std::array<VkDescriptorSetLayoutBinding, 3> sceneBindings{};
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    sceneBindings[2].binding = 2;
    sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneBindings[2].descriptorCount = 1;
    sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
    sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneLayoutInfo.bindingCount = static_cast<uint32_t>(sceneBindings.size());
    sceneLayoutInfo.pBindings = sceneBindings.data();
    vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr, &sceneSetLayout);

    VkDescriptorSetLayoutBinding objectBinding{};
    objectBinding.binding = 0;
    objectBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    objectBinding.descriptorCount = 1;
    objectBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo objectLayoutInfo{};
    objectLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    objectLayoutInfo.bindingCount = 1;
    objectLayoutInfo.pBindings = &objectBinding;
    vkCreateDescriptorSetLayout(context.device, &objectLayoutInfo, nullptr, &objectSetLayout);

    std::array<VkDescriptorSetLayout, 2> litSetLayouts = {sceneSetLayout, objectSetLayout};
    VkPipelineLayoutCreateInfo litPipelineLayoutInfo{};
    litPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    litPipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(litSetLayouts.size());
    litPipelineLayoutInfo.pSetLayouts = litSetLayouts.data();
    vkCreatePipelineLayout(context.device, &litPipelineLayoutInfo, nullptr, &litPipelineLayout);

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstantMarker);

    VkPipelineLayoutCreateInfo unlitPipelineLayoutInfo{};
    unlitPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    unlitPipelineLayoutInfo.setLayoutCount = 0;
    unlitPipelineLayoutInfo.pushConstantRangeCount = 1;
    unlitPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    vkCreatePipelineLayout(context.device, &unlitPipelineLayoutInfo, nullptr, &unlitPipelineLayout);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkVertexInputBindingDescription bindingDesc{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 2> attribDesc{};
    attribDesc[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
    attribDesc[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attribDesc.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyTriangles{};
    inputAssemblyTriangles.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyTriangles.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    VkViewport viewport{0.0f, 0.0f, (float)context.swapchain_extent.width, (float)context.swapchain_extent.height, 0.0f, 1.0f};
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    VkRect2D scissor{{0, 0}, context.swapchain_extent};
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    auto vertLitCode = readSpvFile("shaders/shader.vert.spv");
    auto fragLitCode = readSpvFile("shaders/shader.frag.spv");
    VkShaderModule vertLitModule = buildShaderModule(vertLitCode);
    VkShaderModule fragLitModule = buildShaderModule(fragLitCode);
    VkPipelineShaderStageCreateInfo litStages[2]{};
    litStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertLitModule, "main", nullptr};
    litStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragLitModule, "main", nullptr};

    VkGraphicsPipelineCreateInfo litPipelineInfo{};
    litPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    litPipelineInfo.stageCount = 2;
    litPipelineInfo.pStages = litStages;
    litPipelineInfo.pVertexInputState = &vertexInputInfo;
    litPipelineInfo.pInputAssemblyState = &inputAssemblyTriangles;
    litPipelineInfo.pViewportState = &viewportState;
    litPipelineInfo.pRasterizationState = &rasterizer;
    litPipelineInfo.pMultisampleState = &multisampling;
    litPipelineInfo.pDepthStencilState = &depthStencil;
    litPipelineInfo.pColorBlendState = &colorBlending;
    litPipelineInfo.layout = litPipelineLayout;
    litPipelineInfo.renderPass = context.render_pass;
    vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &litPipelineInfo, nullptr, &litPipeline);
    vkDestroyShaderModule(context.device, fragLitModule, nullptr);
    vkDestroyShaderModule(context.device, vertLitModule, nullptr);

    auto vertUnlitCode = readSpvFile("shaders/unlit.vert.spv");
    auto fragUnlitCode = readSpvFile("shaders/unlit.frag.spv");
    VkShaderModule vertUnlitModule = buildShaderModule(vertUnlitCode);
    VkShaderModule fragUnlitModule = buildShaderModule(fragUnlitCode);
    VkPipelineShaderStageCreateInfo unlitStages[2]{};
    unlitStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertUnlitModule, "main", nullptr};
    unlitStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragUnlitModule, "main", nullptr};

    VkGraphicsPipelineCreateInfo unlitMeshInfo = litPipelineInfo;
    unlitMeshInfo.pStages = unlitStages;
    unlitMeshInfo.layout = unlitPipelineLayout;
    vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &unlitMeshInfo, nullptr, &unlitMeshPipeline);

    VkPipelineVertexInputStateCreateInfo lineVertexInputInfo{};
    lineVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkVertexInputBindingDescription lineBindingDesc{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription lineAttribDesc{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    lineVertexInputInfo.vertexBindingDescriptionCount = 1;
    lineVertexInputInfo.pVertexBindingDescriptions = &lineBindingDesc;
    lineVertexInputInfo.vertexAttributeDescriptionCount = 1;
    lineVertexInputInfo.pVertexAttributeDescriptions = &lineAttribDesc;

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyLines{};
    inputAssemblyLines.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyLines.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineRasterizationStateCreateInfo lineRasterizer = rasterizer;
    lineRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkGraphicsPipelineCreateInfo unlitLineInfo = unlitMeshInfo;
    unlitLineInfo.pVertexInputState = &lineVertexInputInfo;
    unlitLineInfo.pInputAssemblyState = &inputAssemblyLines;
    unlitLineInfo.pRasterizationState = &lineRasterizer;
    vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &unlitLineInfo, nullptr, &unlitLinePipeline);

    vkDestroyShaderModule(context.device, fragUnlitModule, nullptr);
    vkDestroyShaderModule(context.device, vertUnlitModule, nullptr);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2};
    poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 2;
    vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool);

    VkDescriptorSetAllocateInfo sceneAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 1, &sceneSetLayout};
    vkAllocateDescriptorSets(context.device, &sceneAllocInfo, &sceneDescriptorSet);

    VkDescriptorSetAllocateInfo objAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 1, &objectSetLayout};
    vkAllocateDescriptorSets(context.device, &objAllocInfo, &objectDescriptorSet);

    VkDescriptorBufferInfo sceneBufInfo{sceneBuffer, 0, sizeof(SceneUBO)};
    VkDescriptorBufferInfo pointBufInfo{pointLightSSBO, 0, sizeof(GPUPrivatePointLight) * MAX_LIGHTS_CAPACITY};
    VkDescriptorBufferInfo spotBufInfo{spotLightSSBO, 0, sizeof(GPUPrivateSpotLight) * MAX_LIGHTS_CAPACITY};

    std::array<VkWriteDescriptorSet, 4> writes{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sceneBufInfo, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pointBufInfo, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spotBufInfo, nullptr};

    VkDescriptorBufferInfo objBufInfo{objectBuffer, 0, sizeof(ObjectUBO)};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, objectDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &objBufInfo, nullptr};

    vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool initialize() {
    generateSphere(1.0f, 32, 32);

    pointLights.push_back({{-2.5f, 2.0f, 0.0f}, {1.0f, 0.2f, 0.2f}, 3.0f});
    pointLights.push_back({{2.5f, 2.0f, 0.0f}, {0.2f, 0.5f, 1.0f}, 3.0f});
    spotLights.push_back({{0.0f, 3.5f, 2.5f}, {0.0f, -0.8f, -0.6f}, {1.0f, 0.9f, 0.5f}, 5.0f, 15.0f, 25.0f});

    createBuffer(sizeof(Vertex) * sphereVertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexBufferAllocation, sphereVertices.data());
    createBuffer(sizeof(uint32_t) * sphereIndices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexBufferAllocation, sphereIndices.data());

    coneLineMappedData = createBuffer(sizeof(glm::vec3) * 1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, coneLineBuffer, coneLineBufferAllocation);

    sceneMappedData  = createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneBuffer, sceneBufferAllocation);
    objectMappedData = createBuffer(sizeof(ObjectUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, objectBuffer, objectBufferAllocation);
    pointLightSSBOMappedData = createBuffer(sizeof(GPUPrivatePointLight) * MAX_LIGHTS_CAPACITY, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightSSBO, pointLightSSBOAllocation);
    spotLightSSBOMappedData  = createBuffer(sizeof(GPUPrivateSpotLight) * MAX_LIGHTS_CAPACITY, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightSSBO, spotLightSSBOAllocation);

    initPipelinesAndDescriptors();
    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);

    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, objectSetLayout, nullptr);

    vkDestroyPipeline(context.device, litPipeline, nullptr);
    vkDestroyPipeline(context.device, unlitMeshPipeline, nullptr);
    vkDestroyPipeline(context.device, unlitLinePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, litPipelineLayout, nullptr);
    vkDestroyPipelineLayout(context.device, unlitPipelineLayout, nullptr);

    vmaDestroyBuffer(context.allocator, coneLineBuffer, coneLineBufferAllocation);
    vmaDestroyBuffer(context.allocator, pointLightSSBO, pointLightSSBOAllocation);
    vmaDestroyBuffer(context.allocator, spotLightSSBO, spotLightSSBOAllocation);
    vmaDestroyBuffer(context.allocator, sceneBuffer, sceneBufferAllocation);
    vmaDestroyBuffer(context.allocator, objectBuffer, objectBufferAllocation);
    vmaDestroyBuffer(context.allocator, vertexBuffer, vertexBufferAllocation);
    vmaDestroyBuffer(context.allocator, indexBuffer, indexBufferAllocation);
}

void update(double time) {
    ImGui::Begin("Lab 2 Control Panel");

    if (ImGui::CollapsingHeader("Camera Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Camera Pos", &cameraPos.x, -15.0f, 15.0f);
        ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f);
        ImGui::SliderFloat("FOV", &cameraFov, 15.0f, 90.0f);
    }

    if (ImGui::CollapsingHeader("Light Visualization (Grade 5)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show Light Markers (Spheres)", &showLightMarkers);
        ImGui::Checkbox("Show Spotlight Cones (Lines)", &showSpotCones);
    }

    if (ImGui::CollapsingHeader("Directional & Ambient Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Directional Light", &enableDirLight);
        ImGui::Checkbox("Enable Ambient Light", &enableAmbient);
        ImGui::Separator();

        if (enableDirLight) {
            ImGui::SliderFloat3("Sun Direction", &dirLightDir.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("Sun Color", &dirLightColor.x);
            ImGui::SliderFloat("Sun Intensity", &dirLightIntensity, 0.0f, 5.0f);
        }

        if (enableAmbient) {
            ImGui::ColorEdit3("Ambient Color", &ambientColor.x);
        }
    }

    if (ImGui::CollapsingHeader("Object Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Object Pos", &objPosition.x, -5.0f, 5.0f);
        ImGui::SliderFloat3("Object Rotation", &objRotation.x, -180.0f, 180.0f);
        ImGui::SliderFloat3("Object Scale", &objScale.x, 0.1f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Object Material (Blinn-Phong)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", &matAlbedo.x);
        ImGui::ColorEdit3("Specular Color", &matSpecular.x);
        ImGui::SliderFloat("Shininess", &matShininess, 2.0f, 256.0f);
    }

    if (ImGui::CollapsingHeader("Point Lights (SSBO Array)", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("+ Add Point Light") && pointLights.size() < MAX_LIGHTS_CAPACITY) {
            pointLights.push_back({{0.0f, 2.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 2.0f});
        }
        for (int i = 0; i < (int)pointLights.size(); ++i) {
            ImGui::PushID(i);
            char label[32]; sprintf(label, "Point Light #%d", i + 1);
            if (ImGui::TreeNode(label)) {
                ImGui::SliderFloat3("Position", &pointLights[i].position.x, -8.0f, 8.0f);
                ImGui::ColorEdit3("Color", &pointLights[i].color.x);
                ImGui::SliderFloat("Intensity", &pointLights[i].intensity, 0.0f, 10.0f);
                if (ImGui::Button("Delete Light")) {
                    pointLights.erase(pointLights.begin() + i);
                    ImGui::TreePop(); ImGui::PopID(); break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Spotlights (SSBO Array)", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("+ Add Spotlight") && spotLights.size() < MAX_LIGHTS_CAPACITY) {
            spotLights.push_back({{0.0f, 3.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 4.0f, 15.0f, 25.0f});
        }
        for (int j = 0; j < (int)spotLights.size(); ++j) {
            ImGui::PushID(1000 + j);
            char label[32]; sprintf(label, "Spotlight #%d", j + 1);
            if (ImGui::TreeNode(label)) {
                ImGui::SliderFloat3("Position", &spotLights[j].position.x, -8.0f, 8.0f);
                ImGui::SliderFloat3("Direction", &spotLights[j].direction.x, -1.0f, 1.0f);
                ImGui::ColorEdit3("Color", &spotLights[j].color.x);
                ImGui::SliderFloat("Intensity", &spotLights[j].intensity, 0.0f, 15.0f);
                ImGui::SliderFloat("Inner Angle", &spotLights[j].innerAngle, 1.0f, 45.0f);
                ImGui::SliderFloat("Outer Angle", &spotLights[j].outerAngle, spotLights[j].innerAngle, 60.0f);
                if (ImGui::Button("Delete Spotlight")) {
                    spotLights.erase(spotLights.begin() + j);
                    ImGui::TreePop(); ImGui::PopID(); break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    ImGui::End();
    updateConeLines();
}

void render(const graphics::internal::FrameData& fd) {
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.03f, 0.03f, 0.05f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderPassInfo.renderPass = graphics::internal::context.render_pass;
    renderPassInfo.framebuffer = fd.framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = graphics::internal::context.swapchain_extent;
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(fd.command_buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    float aspect = (float)graphics::internal::context.swapchain_extent.width / (float)graphics::internal::context.swapchain_extent.height;
    glm::mat4 proj = glm::perspective(glm::radians(cameraFov), aspect, 0.1f, 100.0f);
    glm::mat4 clipMatrix = glm::mat4(1.0f);
    clipMatrix[1][1] = -1.0f;
    clipMatrix[2][2] = 0.5f;
    clipMatrix[3][2] = 0.5f;
    proj = clipMatrix * proj;

    glm::vec3 front;
    front.x = cosf(glm::radians(cameraYaw)) * cosf(glm::radians(cameraPitch));
    front.y = sinf(glm::radians(cameraPitch));
    front.z = sinf(glm::radians(cameraYaw)) * cosf(glm::radians(cameraPitch));
    front = glm::normalize(front);
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + front, glm::vec3(0.0f, 1.0f, 0.0f));

    SceneUBO sceneUBO{};
    sceneUBO.view = view;
    sceneUBO.proj = proj;
    sceneUBO.viewPos = glm::vec4(cameraPos, 1.0f);
    sceneUBO.ambientColor = enableAmbient ? glm::vec4(ambientColor, 1.0f) : glm::vec4(0.0f);
    sceneUBO.dirLightDir = glm::vec4(glm::normalize(dirLightDir), 0.0f);
    sceneUBO.dirLightColor = enableDirLight ? glm::vec4(dirLightColor * dirLightIntensity, 1.0f) : glm::vec4(0.0f);
    sceneUBO.pointLightCount = static_cast<int>(pointLights.size());
    sceneUBO.spotLightCount = static_cast<int>(spotLights.size());
    memcpy(sceneMappedData, &sceneUBO, sizeof(SceneUBO));

    if (!pointLights.empty() && pointLightSSBOMappedData != nullptr) {
        std::vector<GPUPrivatePointLight> gpuPoints(pointLights.size());
        for (size_t i = 0; i < pointLights.size(); ++i) {
            gpuPoints[i].position = glm::vec4(pointLights[i].position, 1.0f);
            gpuPoints[i].color    = glm::vec4(pointLights[i].color * pointLights[i].intensity, 1.0f);
        }
        memcpy(pointLightSSBOMappedData, gpuPoints.data(), sizeof(GPUPrivatePointLight) * gpuPoints.size());
    }

    if (!spotLights.empty() && spotLightSSBOMappedData != nullptr) {
        std::vector<GPUPrivateSpotLight> gpuSpots(spotLights.size());
        for (size_t j = 0; j < spotLights.size(); ++j) {
            float inCos = cosf(glm::radians(spotLights[j].innerAngle));
            float outCos = cosf(glm::radians(spotLights[j].outerAngle));
            gpuSpots[j].position  = glm::vec4(spotLights[j].position, inCos);
            gpuSpots[j].direction = glm::vec4(glm::normalize(spotLights[j].direction), outCos);
            gpuSpots[j].color     = glm::vec4(spotLights[j].color * spotLights[j].intensity, 1.0f);
        }
        memcpy(spotLightSSBOMappedData, gpuSpots.data(), sizeof(GPUPrivateSpotLight) * gpuSpots.size());
    }

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, objPosition);
    model = glm::rotate(model, glm::radians(objRotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(objRotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(objRotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, objScale);

    ObjectUBO objUBO{};
    objUBO.model = model;
    objUBO.albedo = glm::vec4(matAlbedo, 1.0f);
    objUBO.specular = glm::vec4(matSpecular, matShininess);
    memcpy(objectMappedData, &objUBO, sizeof(ObjectUBO));

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipeline);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipelineLayout, 1, 1, &objectDescriptorSet, 0, nullptr);
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(sphereIndices.size()), 1, 0, 0, 0);

    if (showLightMarkers) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitMeshPipeline);

        for (const auto& light : pointLights) {
            glm::mat4 markerModel = glm::translate(glm::mat4(1.0f), light.position);
            markerModel = glm::scale(markerModel, glm::vec3(0.08f));
            PushConstantMarker pc{proj * view * markerModel, glm::vec4(light.color, 1.0f)};
            vkCmdPushConstants(fd.command_buffer, unlitPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantMarker), &pc);
            vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(sphereIndices.size()), 1, 0, 0, 0);
        }

        for (const auto& spot : spotLights) {
            glm::mat4 markerModel = glm::translate(glm::mat4(1.0f), spot.position);
            markerModel = glm::scale(markerModel, glm::vec3(0.1f));
            PushConstantMarker pc{proj * view * markerModel, glm::vec4(spot.color, 1.0f)};
            vkCmdPushConstants(fd.command_buffer, unlitPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantMarker), &pc);
            vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(sphereIndices.size()), 1, 0, 0, 0);
        }
    }

    if (showSpotCones && coneLineVertexCount > 0) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitLinePipeline);
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &coneLineBuffer, &offset);
        PushConstantMarker pcLines{proj * view, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)};
        vkCmdPushConstants(fd.command_buffer, unlitPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantMarker), &pcLines);
        vkCmdDraw(fd.command_buffer, coneLineVertexCount, 1, 0, 0);
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

}