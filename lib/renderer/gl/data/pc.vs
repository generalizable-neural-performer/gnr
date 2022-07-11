#version 330

layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Color;

out VertexData {
    vec3 Position;
    vec3 Color;
} VertexOut;

uniform mat3 RotMat;
uniform mat4 NormMat;
uniform mat4 ModelMat;
uniform mat4 PerspMat;

void main()
{
    // normalization
    vec3 pos = (NormMat * vec4(a_Position,1.0)).xyz;

    mat3 R = mat3(ModelMat) * RotMat;
    VertexOut.Position = R * pos;
    VertexOut.Color = a_Color;

    gl_Position = PerspMat * ModelMat * vec4(RotMat * pos, 1.0);
}
