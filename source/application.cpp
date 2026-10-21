#include "application.hpp"
#include <imgui.h>
#include <cmath>
#include <cstring>
#include <vk_mem_alloc.h>
#include <fstream>
#include <stdexcept>
#include <array>
#include <string>
#include <glm/gtc/matrix_transform.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace application {

glm::vec3 cameraPos = {0.0f, 0.0f, 5.0f};
float cameraYaw = -90.0f;
float cameraPitch = 0.0f;
float cameraFov = 45.0f;

glm::vec3 dirLightDir = {-0.5f, -1.0f, -0.5f};
glm::vec3 dirLightColor = {1.0f, 1.0f, 1.0f};
float dirLightIntensity = 1.0f;

glm::vec3 ambientColor = {0.15f, 0.15f, 0.15f};

glm::vec3 objPosition = {0.0f, 0.0f, 0.0f};
glm::vec3 objRotation = {0.0f, 0.0f, 0.0f};
glm::vec3 objScale    = {1.0f, 1.0f, 1.0f};

glm::vec3 matAlbedo    = {0.8f, 0.2f, 0.2f};
glm::vec3 matSpecular  = {1.0f, 1.0f, 1.0f};
float matShininess     = 32.0f;

struct alignas(16) SceneUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 viewPos;
    glm::vec4 ambientColor;
    glm::vec4 dirLightDir;
    glm::vec4 dirLightColor;
};

struct alignas(16) ObjectUBO {
    glm::mat4 model;
    glm::vec4 albedo;
    glm::vec4 specular;
};

std::vector<Vertex> sphereVertices;
std::vector<uint32_t> sphereIndices;

VkBuffer vertexBuffer;
VmaAllocation vertexBufferAllocation;

VkBuffer indexBuffer;
VmaAllocation indexBufferAllocation;

VkBuffer sceneBuffer;
VmaAllocation sceneBufferAllocation;
void* sceneMappedData = nullptr;

VkBuffer objectBuffer;
VmaAllocation objectBufferAllocation;
void* objectMappedData = nullptr;

VkDescriptorSetLayout sceneSetLayout;
VkDescriptorSetLayout objectSetLayout;
VkDescriptorPool descriptorPool;

VkDescriptorSet sceneDescriptorSet;
VkDescriptorSet objectDescriptorSet;

VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;

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

std::vector<char> readSpvFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        file.clear();
        file.open("../" + filename, std::ios::ate | std::ios::binary);
    }

    if (!file.is_open()) {
        throw std::runtime_error("not found: " + filename);
    }

    size_t fileSize = (size_t) file.tellg();
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

void initPipeline() {
    auto& context = graphics::internal::context;

    auto vertCode = readSpvFile("shaders/shader.vert.spv");
    auto fragCode = readSpvFile("shaders/shader.frag.spv");

    VkShaderModule vertModule = buildShaderModule(vertCode);
    VkShaderModule fragModule = buildShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertModule;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragModule;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStageInfo, fragStageInfo};

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) context.swapchain_extent.width;
    viewport.height = (float) context.swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = context.swapchain_extent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkDescriptorSetLayoutBinding sceneLayoutBinding{};
    sceneLayoutBinding.binding = 0;
    sceneLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneLayoutBinding.descriptorCount = 1;
    sceneLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
    sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneLayoutInfo.bindingCount = 1;
    sceneLayoutInfo.pBindings = &sceneLayoutBinding;
    vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr, &sceneSetLayout);

    VkDescriptorSetLayoutBinding objectLayoutBinding{};
    objectLayoutBinding.binding = 0;
    objectLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    objectLayoutBinding.descriptorCount = 1;
    objectLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo objectLayoutInfo{};
    objectLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    objectLayoutInfo.bindingCount = 1;
    objectLayoutInfo.pBindings = &objectLayoutBinding;
    vkCreateDescriptorSetLayout(context.device, &objectLayoutInfo, nullptr, &objectSetLayout);

    std::array<VkDescriptorSetLayout, 2> setLayouts = {sceneSetLayout, objectSetLayout};

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();

    vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = context.render_pass;
    pipelineInfo.subpass = 0;

    vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);

    vkDestroyShaderModule(context.device, fragModule, nullptr);
    vkDestroyShaderModule(context.device, vertModule, nullptr);
}

void initDescriptors() {
    auto& context = graphics::internal::context;

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 2;

    vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool);

    VkDescriptorSetAllocateInfo sceneAllocInfo{};
    sceneAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sceneAllocInfo.descriptorPool = descriptorPool;
    sceneAllocInfo.descriptorSetCount = 1;
    sceneAllocInfo.pSetLayouts = &sceneSetLayout;
    vkAllocateDescriptorSets(context.device, &sceneAllocInfo, &sceneDescriptorSet);

    VkDescriptorSetAllocateInfo objAllocInfo{};
    objAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    objAllocInfo.descriptorPool = descriptorPool;
    objAllocInfo.descriptorSetCount = 1;
    objAllocInfo.pSetLayouts = &objectSetLayout;
    vkAllocateDescriptorSets(context.device, &objAllocInfo, &objectDescriptorSet);

    VkDescriptorBufferInfo sceneBufferInfo{};
    sceneBufferInfo.buffer = sceneBuffer;
    sceneBufferInfo.offset = 0;
    sceneBufferInfo.range = sizeof(SceneUBO);

    VkWriteDescriptorSet sceneWrite{};
    sceneWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    sceneWrite.dstSet = sceneDescriptorSet;
    sceneWrite.dstBinding = 0;
    sceneWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneWrite.descriptorCount = 1;
    sceneWrite.pBufferInfo = &sceneBufferInfo;

    VkDescriptorBufferInfo objBufferInfo{};
    objBufferInfo.buffer = objectBuffer;
    objBufferInfo.offset = 0;
    objBufferInfo.range = sizeof(ObjectUBO);

    VkWriteDescriptorSet objWrite{};
    objWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    objWrite.dstSet = objectDescriptorSet;
    objWrite.dstBinding = 0;
    objWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    objWrite.descriptorCount = 1;
    objWrite.pBufferInfo = &objBufferInfo;

    std::array<VkWriteDescriptorSet, 2> descriptorWrites = {sceneWrite, objWrite};
    vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

bool initialize() {
    generateSphere(1.0f, 32, 32);

    VkDeviceSize vertexBufferSize = sizeof(Vertex) * sphereVertices.size();
    createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexBufferAllocation, sphereVertices.data());

    VkDeviceSize indexBufferSize = sizeof(uint32_t) * sphereIndices.size();
    createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexBufferAllocation, sphereIndices.data());

    sceneMappedData  = createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneBuffer, sceneBufferAllocation);
    objectMappedData = createBuffer(sizeof(ObjectUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, objectBuffer, objectBufferAllocation);

    initPipeline();
    initDescriptors();

    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);

    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, objectSetLayout, nullptr);

    vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);

    vmaDestroyBuffer(context.allocator, sceneBuffer, sceneBufferAllocation);
    vmaDestroyBuffer(context.allocator, objectBuffer, objectBufferAllocation);
    vmaDestroyBuffer(context.allocator, vertexBuffer, vertexBufferAllocation);
    vmaDestroyBuffer(context.allocator, indexBuffer, indexBufferAllocation);
}

void update(double time) {
    ImGui::Begin("Controls");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Camera Pos", &cameraPos.x, -10.0f, 10.0f);
        ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f);
        ImGui::SliderFloat("FOV", &cameraFov, 10.0f, 90.0f);
    }

    if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Light Dir", &dirLightDir.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Light Color", &dirLightColor.x);
        ImGui::SliderFloat("Intensity", &dirLightIntensity, 0.0f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Ambient Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient Color", &ambientColor.x);
    }

    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", &matAlbedo.x);
        ImGui::ColorEdit3("Specular", &matSpecular.x);
        ImGui::SliderFloat("Shininess", &matShininess, 2.0f, 256.0f);
    }

    if (ImGui::CollapsingHeader("Object Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", &objPosition.x, -5.0f, 5.0f);
        ImGui::SliderFloat3("Rotation", &objRotation.x, -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale", &objScale.x, 0.1f, 3.0f);
    }

    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.05f, 0.05f, 0.05f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = graphics::internal::context.render_pass;
    renderPassInfo.framebuffer = fd.framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = graphics::internal::context.swapchain_extent;
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(fd.command_buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    float aspect = (float)graphics::internal::context.swapchain_extent.width / (float)graphics::internal::context.swapchain_extent.height;

    glm::mat4 proj = glm::perspective(glm::radians(cameraFov), aspect, 0.1f, 100.0f);
    glm::mat4 clipMatrix = glm::mat4(1.0f);
    clipMatrix[1][1] = -1.0f;
    clipMatrix[2][2] =  0.5f;
    clipMatrix[3][2] =  0.5f;
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
    sceneUBO.ambientColor = glm::vec4(ambientColor, 1.0f);
    sceneUBO.dirLightDir = glm::vec4(glm::normalize(dirLightDir), 0.0f);
    sceneUBO.dirLightColor = glm::vec4(dirLightColor * dirLightIntensity, 1.0f);
    memcpy(sceneMappedData, &sceneUBO, sizeof(SceneUBO));

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

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &objectDescriptorSet, 0, nullptr);

    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(sphereIndices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

}