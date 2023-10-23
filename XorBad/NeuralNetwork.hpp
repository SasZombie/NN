#pragma once

#include <iostream>
#include <assert.h>
#include <random>
#include <memory>

#define mat_at(m, i, j)  (m)->es[(i) * (m)->stride + (j)]
#define mat_at_non(m, i, j)  (m).es[(i) * (m).stride + (j)]


namespace nn
{
    class Mat
    {
    private:
        size_t rows, cols, stride;
        std::shared_ptr<float[]> es;
        float getRandom() const noexcept;
        //float at(const size_t i, const size_t j) const noexcept;
        float sig(const float x) noexcept;

    public:
        Mat() = default;
        Mat(size_t n_rows, size_t n_cols)
            : rows(n_rows), cols(n_cols), stride(n_cols), es(std::make_shared<float[]>(n_rows * n_cols))
        {
           
        }
        Mat(size_t n_rows, size_t n_cols, size_t n_stride)
            : rows(n_rows), cols(n_cols), stride(n_stride), es(std::make_shared<float[]>(n_rows * n_cols))
        {
           
        }

        Mat(size_t n_rows, size_t n_cols, size_t n_stride, const float *n_es)
            : rows(n_rows), cols(n_cols), stride(n_stride), es(std::make_shared<float[]>(n_rows * n_cols))
        {
            std::copy(n_es, n_es + rows * stride, es.get());
        }


        ~Mat() = default;
        
        void setEs(const float *n_es);
        size_t getRows() const noexcept;
        size_t getCols() const noexcept;
        void alloc(const size_t n_rows, const size_t n_cols);
        void alloc(const size_t n_rows, const size_t n_cols, const size_t n_stride);
        Mat matRow(size_t row) const;
        void rand(const float low = 0, const float max = 1) noexcept;
        void fill(const float x) noexcept;
        void dot(const Mat a, const Mat b);
        void sum(const Mat a);
        void print(const std::string& name = "") const noexcept; 
        void clear() noexcept; 
        void setAt(const size_t i, const size_t j, const float number) noexcept;
        float getAt(const size_t i, const size_t j) const noexcept;
        void apply_sigmoid() noexcept;

    };    


    class NN
    {
    private:

        size_t count;
        nn::Mat *ws;
        nn::Mat *bs;
        nn::Mat *as; //Count + 1 
    public:

        #define NN_INPUT(nn) (nn).as[0]
        #define NN_OUTPUT(nn) (nn).as[nn.count]

        NN() = default;
        NN(size_t *arch, size_t arch_count);
        void print();
        void alloc(size_t *arch, size_t arch_count);
        void rand(const float low = 0, const float max = 1);
   
        void forward() noexcept;
        void fineDiff(NN &grad, const float eps, const Mat& ti, const Mat& to);
        void learn(const NN &grad, float rate);
        float cost(const Mat& ti, const Mat& to);

     
        //Getters And Setters >_<
        nn::Mat getInput() const noexcept;
        nn::Mat getOutput() const noexcept;
        void setInput(const nn::Mat& m) const noexcept;
        void setOutput(const nn::Mat& m) const noexcept;
        size_t getCount() const noexcept;
        float getAtWs(const size_t i, const size_t j, const size_t k) const noexcept;
        void setAtWs(const size_t i, const size_t j, const size_t k, const float number) noexcept;
        float getAtBs(const size_t i, const size_t j, const size_t k) const noexcept;
        void setAtBs(const size_t i, const size_t j, const size_t k, const float number) noexcept;
        float getAtAs(const size_t i, const size_t j, const size_t k) const noexcept;
        void setAtAs(const size_t i, const size_t j, const size_t k, const float number) noexcept;
             

        ~NN() = default;
    };    
 
    
} // namespace nn

