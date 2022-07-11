#version 330


in VertexData {
    vec3 Position;
    vec3 Color;
} VertexIn;

layout (location = 0) out vec4 FragColor;

void main()
{
    FragColor = vec4(VertexIn.Color, 1.0f);
}