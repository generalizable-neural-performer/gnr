#version 330


in VertexData {
    vec3 Position;
    vec3 ModelNormal;
} VertexIn;

layout (location = 0) out vec4 FragColor;

uniform vec3 SHCoeffs[9];

vec4 gammaCorrection(vec4 vec, float g)
{
    return vec4(pow(vec.x, 1.0/g), pow(vec.y, 1.0/g), pow(vec.z, 1.0/g), vec.w);
}

void evaluateH(vec3 n, out float H[9])
{
    float c1 = 0.429043, c2 = 0.511664,
        c3 = 0.743125, c4 = 0.886227, c5 = 0.247708;

    H[0] = c4;
    H[1] = 2.0 * c2 * n[1];
    H[2] = 2.0 * c2 * n[2];
    H[3] = 2.0 * c2 * n[0];
    H[4] = 2.0 * c1 * n[0] * n[1];
    H[5] = 2.0 * c1 * n[1] * n[2];
    H[6] = c3 * n[2] * n[2] - c5;
    H[7] = 2.0 * c1 * n[2] * n[0];
    H[8] = c1 * (n[0] * n[0] - n[1] * n[1]);
}

vec3 evaluateLightingModel(vec3 normal)
{
    float H[9];
    evaluateH(normal, H);
    vec3 res = vec3(0.0);
    for (int i = 0; i < 9; i++) {
        res += H[i] * SHCoeffs[i];
    }
    return res;
}

void main()
{
    // vec3 light_pos = vec3(0.0, 0.0, 1.0);
    // vec3 ambient = vec3(1.0, 0.65, 0.0);

    // vec3 norm = normalize(VertexIn.ModelNormal);
    // vec3 lightDir = normalize(light_pos - VertexIn.Position);
    // float diff = max(dot(norm, lightDir), 0.0f);
    // vec3 diffuse = diff * ambient;

    // FragColor = vec4(ambient + diffuse , 1.0f);
    vec3 nC = normalize(VertexIn.ModelNormal);
    vec4 FragAlbedo = vec4(1.0, 0.65, 0.0, 1.0);
    vec4 FragShading = vec4(evaluateLightingModel(nC), 1.0f);

    FragShading = gammaCorrection(FragShading, 2.2);
    FragColor = clamp(FragAlbedo * FragShading, 0.0, 1.0);
}