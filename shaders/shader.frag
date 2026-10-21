#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 viewPos;
    vec4 ambientColor;
    vec4 dirLightDir;
    vec4 dirLightColor;
} scene;

layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
} object;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.viewPos.xyz - fragWorldPos);

    vec3 ambient = scene.ambientColor.rgb * object.albedo.rgb;

    vec3 L = normalize(-scene.dirLightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * scene.dirLightColor.rgb * object.albedo.rgb;

    vec3 H = normalize(L + V);
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(N, H), 0.0), object.specular.w);
    }
    vec3 specularHighlight = spec * scene.dirLightColor.rgb * object.specular.rgb;

    vec3 result = ambient + diffuse + specularHighlight;
    outColor = vec4(result, 1.0);
}