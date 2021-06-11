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

#include <helper_cuda.h>


#include "Matrix.h"
#include "cudacheck.h"
#include "DataBlock.h"
#include "Simulator.h"
#include "MyResource.h"
#include "Detector.h"


#include "vendor/imgui/imgui.h"
#include "vendor/imgui/imgui_impl_glfw.h"
#include "vendor/imgui/imgui_impl_opengl3.h"

#define window_width 640 * 2
#define window_height 640
#define TILE_SIZE 16





int main()
{

    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);


    window = glfwCreateWindow(window_width, window_height, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);
    

    if (glewInit() != GLEW_OK)
        std::cout << "GLEW ini failed!" << std::endl;

    std::cout << glGetString(GL_VERSION) << std::endl;

    GLCall(glEnable(GL_BLEND));
    GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));


    {

        cudaDeviceProp prop;
        int dev = 0;



        CUDACheck(cudaGetDeviceProperties(&prop, dev));
        std::cout << prop.name << std::endl;
        CUDACheck(cudaSetDevice(dev));


        Translator world(0.0, 0.0, 200.0, 200.0);
        world.SetSpace(0.1, 0.1, 0.0707106);
        world.SetWall(16);
        world.Initialize();

        Translator camera(0.0, 0.0, 200.0, 200.0);
        camera.SetSpace(0.1, 0.1);
        camera.SetWall();
        camera.Initialize();



        Matrix hz(world.M, world.N);
        Matrix ex(world.M, world.N);
        Matrix ey(world.M, world.N);
        Matrix Chzh(world.M_const, world.N_const);
        Matrix Chzex(world.M_const, world.N_const);
        Matrix Chzey(world.M_const, world.N_const);
        Matrix Cexe(world.M_const, world.N_const);
        Matrix Cexhz(world.M_const, world.N_const);
        Matrix Ceye(world.M_const, world.N_const);
        Matrix Ceyhz(world.M_const, world.N_const);

        //hz.Circle(hz.GetM() / 4, hz.GetN() / 4, 6, 1);
        //hz.Circle(hz.GetM() * 3 / 4, hz.GetN() * 3 / 4, 6, 1);
        hz.Circle(hz.GetM() / 2, hz.GetN() / 2, 6, 1);

        Chzh.SetValue(1.0f);
        Chzex.SetValue(0.00187695748f);
        Chzey.SetValue(-0.00187695748f);
        Cexe.SetValue(1.0f);
        Cexhz.SetValue(266.3885593f);
        Ceye.SetValue(1.0f);
        Ceyhz.SetValue(-266.3885593f);

        DataBlock D_Hz(hz);
        DataBlock D_Ex(ex);
        DataBlock D_Ey(ey);
        DataBlock D_Chzh(Chzh);
        DataBlock D_Chzex(Chzex);
        DataBlock D_Chzey(Chzey);
        DataBlock D_Cexe(Cexe);
        DataBlock D_Cexhz(Cexhz);
        DataBlock D_Ceye(Ceye);
        DataBlock D_Ceyhz(Ceyhz);



        D_Hz.Initialize();
        D_Ex.Initialize();
        D_Ey.Initialize();
        D_Chzh.Initialize();
        D_Chzex.Initialize();
        D_Chzey.Initialize();
        D_Cexe.Initialize();
        D_Cexhz.Initialize();
        D_Ceye.Initialize();
        D_Ceyhz.Initialize();




        Simulator simulator(D_Hz, D_Ex, D_Ey,
            D_Chzh, D_Chzex, D_Chzey,
            D_Cexe, D_Cexhz,
            D_Ceye, D_Ceyhz);

        Detector detector(D_Hz, camera, world);

        {
            int temp[2] = { world.M, world.M_const };
            simulator.SetConst(temp, 2);
        }
        





        float positions[] =
        {
            0.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
        };
        unsigned int indices[] =
        {
            0, 1, 2,
            2, 3, 0
        };



        Renderer renderer;
        Shader shader("res/shaders/Basic.shader");

        VertexArray va;

        VertexBuffer vb(positions, 4 * 4 * sizeof(float));
        VertexBufferLayout layout;
        layout.Push<float>(2);
        layout.Push<float>(2);

        va.AddBuffer(vb, layout);

        //Texture texture("res/textures/Minato_Aqua_Portrait.png");
        Texture tex;

        IndexBuffer ib(indices, 6);



        va.Unbind();
        vb.Unbind();
        ib.Unbind();
        shader.Unbind();



        int cli = 0;

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
        bool isfield = false;
        bool ispause = false;
        bool vsync = true;
        ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        float fb = 0.0f;
        float fs = 1.0f;
        float timeM = 0.0f;
        float timeE = 0.0f;
        float timeDe = 0.0f;
        float timeDr = 0.0f;

        float time_step = world.dt;
        float duration = 0;
        float time_cost = 0;
        

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window))
        {
            /* Render here */
            renderer.Clear();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            {

                

                ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

                ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
                ImGui::Checkbox("Field Mode.", &isfield); 
                ImGui::SameLine();
                ImGui::Text("cT is %3.1f um\n", duration);
                ImGui::Checkbox("Pause.", &ispause);
                ImGui::SameLine();
                ImGui::Text("Time cost %3.1f s", time_cost);
                ImGui::Checkbox("Vsync On", &vsync);
                // Edit bools storing our window open/close state
                //ImGui::Checkbox("Another Window", &show_another_window);

                ImGui::SliderFloat("contrust / 1", &fb, 0.0f, 0.5f);
                ImGui::SliderFloat("contrust / 0.001", &fs, 0.1f, 1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
                ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

                //ImGui::Button("FieldMode")                           // Buttons return true when clicked (most widgets return true when edited/activated)
                   
                //ImGui::SameLine();
                ImGui::Text("Time to update magnetic %3.1f ms\n", timeM);
                ImGui::Text("Time to update electric %3.1f ms\n", timeE);
                ImGui::Text("Time to detect %3.1f ms\n", timeDe);
                ImGui::Text("Time to draw %3.1f ms\n", timeDr);

                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
                ImGui::End();
            }
            glfwSwapInterval((int)vsync);
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


            shader.Bind();


            if (!ispause)
            {

                {GPUTimer t(timeM);
                simulator.UpdateMagneticFields(); }
                simulator.Sync();
                {GPUTimer t(timeE);
                simulator.UpdateElectricFields(); }
                simulator.Sync();
            

                duration += time_step;
                time_cost += 1.0f / ImGui::GetIO().Framerate;


                std::cout << cli++ << "\n";

            }
                detector.SetPowerMode(!isfield);
                {GPUTimer t(timeDe);
                detector.Detect(); }
                detector.Sync();
                {GPUTimer t(timeDr);
                detector.DrawPixel(fb + fs * 0.001); }
                detector.Sync();






                tex.ReadRGBA(detector.GetPixel(), 0);

            
            shader.SetUniform1i("u_Texture", 0);
            
            

            
            renderer.Draw(va, ib, shader);



            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
        }




    }


    glfwTerminate();
    return 0;
}