#include "NeuralNetwork.hpp"


float td[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 1
};

int main()
{

    size_t arch[] = {2, 2, 1};

    nn::NN nn(arch, 3);
    nn::NN grad(arch, 3);

    nn.rand();
    
    size_t stride = 3;

    size_t n = sizeof(td)/sizeof(td[0])/stride;

    nn::Mat ti(n, 2, stride, td);

    nn::Mat to(n, 1, stride, td+2);
    

    float eps = 1e-1;
    float rate = 1e-1;



#if 1

    std::cout << nn.cost(ti, to) << '\n';
    for(size_t i = 0; i <  20; ++i)
    {
        nn.fineDiff(grad, eps, ti, to);
        nn.learn(grad, rate);
        std::cout << nn.cost(ti, to) << '\n';

    }
    
#endif
}