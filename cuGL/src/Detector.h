#include "MyResource.h"
#include "DataBlock.h"


struct Translator
{
    double minx, miny, maxx, maxy; // unit : micrometer
    double dx, dy, dt;
    int wall, M, N;
    int M_const, N_const;
    Translator(double minx, double miny, double maxx, double maxy)
    {
        this->minx = minx;
        this->miny = miny;
        this->maxx = maxx;
        this->maxy = maxy;
    }
    void SetSpace(double dx, double dy, double dt = 0.0)
    {
        this->dx = dx;
        this->dy = dy;
        this->dt = dt;
    }
    void SetWall(int wall = 0)
    {
        this->wall = wall;
    }
    void Initialize()
    {
        M = (int)floor((maxx - minx) / dx) + 1;
        M = ((M % 16) == 0) ? M : (((M / 16) + 1) * 16);
        maxx = minx + dx * (double)M;
        M_const = M;
        M += 2 * wall;
        N = (int)floor((maxy - miny) / dy) + 1;
        N = ((N % 16) == 0) ? N : (((N / 16) + 1) * 16);
        maxy = miny + dy * (double)N;
        N_const = N;
        N += 2 * wall;
    }


};

void DetectorDetectInterface(float* field, float* intensity, int M_field, int M_camera,
    dim3 grids, dim3 threads, bool ispower);

void DetectorDrawPixelInterface(uchar4* ptr, float* intensity, float contrust,
 dim3 grids, dim3 threads, bool ispower);

class Detector
{
private:
    Translator m_Camera, m_World;
	float* m_Fields;
	float* m_Intensity;
    PixelBuffer m_PixelBuffer;
	MyResource m_R;
    dim3 m_DetectGrids, m_DetectThreads;
    dim3 m_DrawGrids, m_DrawThreads;
    bool m_ispower;
public:
    Detector(DataBlock& D_Fields, Translator& camera, Translator& world);
    ~Detector();

    void SetPowerMode(bool ispower);
 
    void Detect();
    void DrawPixel(float contrust);

    void Sync() const;
    PixelBuffer& GetPixel()  { return m_PixelBuffer; }

private:
    void SetIntensity();
   
  

	

};
