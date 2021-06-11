#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "Renderer.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "VertexBufferLayout.h"
#include "Shader.h"
#include "Texture.h"
#include "PixelBuffer.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#define window_width 640
#define window_height 480

__global__ void kernel(uchar4* ptr, int frequency)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x / (float)window_width - 0.5f;
    float fy = y / (float)window_height - 0.5f;
    unsigned char green = 128 - 127 * sin(abs(fx * frequency) - abs(fy * frequency));

    ptr[offset].x = 0;
    ptr[offset].y = green;
    ptr[offset].z = 0;
    ptr[offset].w = 255;
}


void drawpic(PixelBuffer& pixelbuffer, cudaGraphicsResource*& resource, int frequency)
{
    cudaGraphicsGLRegisterBuffer(&resource, pixelbuffer.GetID(), cudaGraphicsMapFlagsNone); // Register buffer as a graphic source
    uchar4* devPtr;
    size_t size;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

    dim3 grids(window_width / 16, window_height / 16);
    dim3 threads(16, 16);
    
    kernel << <grids, threads >> > (devPtr, frequency);

    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &resource, 0);

}



int main(void)
{
    
    GLFWwindow* window;
    cudaGraphicsResource* resource;

    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);

    cudaGLSetGLDevice(dev);

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(window_width, window_height, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK)
        std::cout << "GLEW ini failed!" << std::endl;

    std::cout << glGetString(GL_VERSION) << std::endl;

    {float positions[] =
        {
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            -1.0f, 1.0f, 0.0f, 1.0f,
    };


    unsigned int indices[] =
    {
        0, 1, 2,
        2, 3, 0
    };

    GLCall(glEnable(GL_BLEND));
    GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

    Renderer renderer;
    Shader shader("res/shaders/Basic.shader");

    VertexArray va;

    VertexBuffer vb(positions, 4 * 4 * sizeof(float));
    VertexBufferLayout layout;
    layout.Push<float>(2);
    layout.Push<float>(2);

    va.AddBuffer(vb, layout);

    Texture texture("res/textures/Minato_Aqua_Portrait.png");
    Texture tex;

    IndexBuffer ib(indices, 6);
    PixelBuffer pb(nullptr, window_width, window_height, 4);

    va.Unbind();
    vb.Unbind();
    ib.Unbind();
    shader.Unbind();

    
   
    int frequency = 5;
    int increment = 1;


    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        renderer.Clear();
        shader.Bind();

        drawpic(pb, resource, frequency);
        tex.ReadRGBA(pb, 0);


        shader.SetUniform1i("u_Texture", 0);
        renderer.Draw(va, ib, shader);

        if (frequency > 100)
            increment = -1;
        else if (frequency < 5)
            increment = 1;

        frequency += increment;
        
        
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    }


    glfwTerminate();
    return 0;
}