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

// Состояние интерфейса для дополнительных заданий
bool isOrthographic = false;                
glm::vec3 objPosition = {0.0f, 0.0f, 0.0f};
glm::vec3 objRotation = {0.0f, 0.0f, 0.0f}; 
glm::vec3 objScale = {1.0f, 1.0f, 1.0f};    

bool isAnimating = false;                   
float animSpeed = 1.0f;                     
float animRadius = 2.0f;                    
float currentAnimTime = 0.0f;               

glm::vec3 objColor = {1.0f, 1.0f, 1.0f};

struct PushConstants {
    glm::mat4 mvp;
    glm::vec4 colorMultiplier;
};

std::vector<Vertex> sphereVertices;
std::vector<uint32_t> sphereIndices;

VkBuffer vertexBuffer;
VmaAllocation vertexBufferAllocation;

VkBuffer indexBuffer;
VmaAllocation indexBufferAllocation;

VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, void* data) {
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
}


void generateSphere(float radius, int sectorCount, int stackCount) {  //горизонтали вертикали
    sphereVertices.clear();
    sphereIndices.clear();

    float sectorStep = 2.0f * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;

    for (int i = 0; i <= stackCount; ++i) {
        float stackAngle = M_PI / 2.0f - i * stackStep; //широта от +90 до -=90
        float y = radius * sinf(stackAngle);
        float xy = radius * cosf(stackAngle);

        for (int j = 0; j <= sectorCount; ++j) {
            float sectorAngle = j * sectorStep;  //долгота от 0 до 360
            float x = xy * cosf(sectorAngle);
            float z = xy * sinf(sectorAngle);

            glm::vec3 color = glm::vec3(
                (x / radius + 1.0f) / 2.0f,
                (y / radius + 1.0f) / 2.0f,
                (z / radius + 1.0f) / 2.0f
            );
            

            sphereVertices.push_back({glm::vec3(x, y, z), color});
        }
    }

    for (int i = 0; i < stackCount; ++i) {
        int k1 = i * (sectorCount + 1);
        int k2 = k1 + sectorCount + 1;

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {    //цвет каждой вершины
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
        throw std::runtime_error("ФАЙЛ ШЕЙДЕРА НЕ НАЙДЕН: " + filename);
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
    auto vertCode = readSpvFile("shaders/shader.vert.spv");
    auto fragCode = readSpvFile("shaders/shader.frag.spv");

    VkShaderModule vertModule = buildShaderModule(vertCode);
    VkShaderModule fragModule = buildShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStageInfo{}; //шейдеры на вершины
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertModule;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{}; //шейдеры на фрагменты
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
    attributeDescriptions[0].offset = 0;

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = sizeof(glm::vec3);           //как gpu забирает инфу

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;    //3 индекса = треугольник
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) graphics::internal::context.swapchain_extent.width;
    viewport.height = (float) graphics::internal::context.swapchain_extent.height;  //создание вьюпорта для exe
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = graphics::internal::context.swapchain_extent;

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
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;   //если line то будет просто карскс типа
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

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    vkCreatePipelineLayout(graphics::internal::context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

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
    pipelineInfo.renderPass = graphics::internal::context.render_pass;
    pipelineInfo.subpass = 0;

    vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);

    vkDestroyShaderModule(graphics::internal::context.device, fragModule, nullptr);
    vkDestroyShaderModule(graphics::internal::context.device, vertModule, nullptr);
}

bool initialize() {
    generateSphere(1.0f, 10, 10);

    VkDeviceSize vertexBufferSize = sizeof(Vertex) * sphereVertices.size();
    createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexBufferAllocation, sphereVertices.data());

    VkDeviceSize indexBufferSize = sizeof(uint32_t) * sphereIndices.size();
    createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexBufferAllocation, sphereIndices.data());

    initPipeline();

    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);

    vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);

    vmaDestroyBuffer(context.allocator, vertexBuffer, vertexBufferAllocation);
    vmaDestroyBuffer(context.allocator, indexBuffer, indexBufferAllocation);
}

void update(double time) {
    // Вычисление дельты времени для независимой от FPS анимации
    static double lastTime = 0.0;
    if (lastTime == 0.0) lastTime = time;
    float deltaTime = static_cast<float>(time - lastTime);
    lastTime = time;

    ImGui::Begin("Controls");

    // Переключение матрицы проекции
    ImGui::Text("Projection");
    if (ImGui::RadioButton("Perspective", !isOrthographic)) isOrthographic = false;
    ImGui::SameLine();
    if (ImGui::RadioButton("Orthographic", isOrthographic)) isOrthographic = true;
    ImGui::Separator();

    //Элементы интерфейса для трансформаций
    ImGui::Text("Transformations");
    ImGui::SliderFloat3("Position", &objPosition.x, -5.0f, 5.0f);
    ImGui::SliderFloat3("Rotation", &objRotation.x, -180.0f, 180.0f);
    ImGui::SliderFloat3("Scale", &objScale.x, 0.1f, 5.0f);
    ImGui::Separator();

    // Движение по сложной траектории
    ImGui::Text("Animation");
    ImGui::Checkbox("Play / Pause", &isAnimating);
    ImGui::SliderFloat("Speed", &animSpeed, 0.1f, 5.0f);
    ImGui::SliderFloat("Radius", &animRadius, 0.5f, 5.0f);
    
    if (isAnimating) {
        currentAnimTime += deltaTime * animSpeed;
        // Траектория + вращение
        objPosition.x = sin(currentAnimTime) * animRadius;
        objPosition.y = sin(currentAnimTime * 2.0f) * (animRadius * 0.5f);
        objPosition.z = cos(currentAnimTime) * (animRadius * 0.5f);
        
        objRotation.x += deltaTime * 50.0f * animSpeed;
        objRotation.y += deltaTime * 30.0f * animSpeed;
    }
    ImGui::Separator();

    // Элемент интерфейса для изменения цвета
    ImGui::Text("Color");
    ImGui::ColorEdit3("Object Color", &objColor.x);

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
    
    // Применение выбранной проекции
    glm::mat4 proj;
    if (isOrthographic) {
        float orthoSize = 3.0f;
        proj = glm::ortho(-orthoSize * aspect, orthoSize * aspect, -orthoSize, orthoSize, 0.1f, 100.0f);
    } else {
        proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    }

    glm::mat4 clipMatrix = glm::mat4(1.0f);
    clipMatrix[1][1] = -1.0f;
    clipMatrix[2][2] =  0.5f;
    clipMatrix[3][2] =  0.5f;
    
    proj = clipMatrix * proj;

    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    
    // Сборка матрицы модели на основе позиции, вращения и масштаба, аффинные
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, objPosition);
    model = glm::rotate(model, glm::radians(objRotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(objRotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(objRotation.z), glm::vec3(0.0f, 0.0f, 1.0f));   
    model = glm::scale(model, objScale);

    // Заполнение структуры PushConstants
    PushConstants pcs{};
    pcs.mvp = proj * view * model;    //модель в мир, камера на мир, проеция
    pcs.colorMultiplier = glm::vec4(objColor, 1.0f);

    // Отправка матриц и цвета в шейдер
    vkCmdPushConstants(fd.command_buffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pcs);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(sphereIndices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application