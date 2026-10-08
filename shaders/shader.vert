#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 colorMultiplier;
} pcs;

void main() {
    gl_Position = pcs.mvp * vec4(inPosition, 1.0);
    // Задание 5: Умножение процедурного цвета вершин на выбранный в интерфейсе цвет
    fragColor = inColor * pcs.colorMultiplier.rgb;
}