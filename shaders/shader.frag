#version 450

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
};

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

layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
} object;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.viewPos.xyz - fragWorldPos);

    vec3 accumulatedLight = scene.ambientColor.rgb * object.albedo.rgb;

    vec3 L_dir = normalize(-scene.dirLightDir.xyz);
    float diff_dir = max(dot(N, L_dir), 0.0);
    vec3 diffuse_dir = diff_dir * scene.dirLightColor.rgb * object.albedo.rgb;
    vec3 H_dir = normalize(L_dir + V);
    float spec_dir = (diff_dir > 0.0) ? pow(max(dot(N, H_dir), 0.0), object.specular.w) : 0.0;
    vec3 specular_dir = spec_dir * scene.dirLightColor.rgb * object.specular.rgb;
    accumulatedLight += (diffuse_dir + specular_dir);

    for (int i = 0; i < scene.pointLightCount; ++i) {
        vec3 lightVec = pointLights[i].position.xyz - fragWorldPos;
        float dist = length(lightVec);
        float attenuation = 1.0 / max(dist * dist, 0.001);
        vec3 L = normalize(lightVec);
        float diff = max(dot(N, L), 0.0);
        vec3 diffuse = diff * pointLights[i].color.rgb * object.albedo.rgb;
        vec3 H = normalize(L + V);
        float spec = (diff > 0.0) ? pow(max(dot(N, H), 0.0), object.specular.w) : 0.0;
        vec3 specular = spec * pointLights[i].color.rgb * object.specular.rgb;
        accumulatedLight += (diffuse + specular) * attenuation;
    }

    for (int j = 0; j < scene.spotLightCount; ++j) {
        vec3 lightVec = spotLights[j].position.xyz - fragWorldPos;
        float dist = length(lightVec);
        float attenuation = 1.0 / max(dist * dist, 0.001);
        vec3 L = normalize(lightVec);
        float theta = dot(L, normalize(-spotLights[j].direction.xyz));
        float innerCutOff = spotLights[j].position.w;
        float outerCutOff = spotLights[j].direction.w;
        float epsilon = innerCutOff - outerCutOff;
        float spotFactor = clamp((theta - outerCutOff) / max(epsilon, 0.0001), 0.0, 1.0);
        float diff = max(dot(N, L), 0.0);
        vec3 diffuse = diff * spotLights[j].color.rgb * object.albedo.rgb;
        vec3 H = normalize(L + V);
        float spec = (diff > 0.0) ? pow(max(dot(N, H), 0.0), object.specular.w) : 0.0;
        vec3 specular = spec * spotLights[j].color.rgb * object.specular.rgb;
        accumulatedLight += (diffuse + specular) * attenuation * spotFactor;
    }

    outColor = vec4(accumulatedLight, 1.0);
}