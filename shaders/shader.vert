#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 viewPos;
    vec4 ambientColor;
    vec4 dirLightDir;
    vec4 dirLightColor;
    int pointLightCount;
    int spotLightCount;
    int pad1;
    int pad2;
} scene;

layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
} object;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;

void main() {
    vec4 worldPos = object.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(object.model))) * inNormal;
    gl_Position = scene.proj * scene.view * worldPos;
}