#include "NeuralNetwork.hpp"

void nn::Mat::setEs(const float* n_es)
{
    std::copy(n_es, n_es + rows * stride, es.get());
}

size_t nn::Mat::getRows() const noexcept
{
    return rows;
}

size_t nn::Mat::getCols() const noexcept
{
    return cols;
}

void nn::Mat::alloc(const size_t n_rows, const size_t n_cols)
{
    this->rows = n_rows;
    this->cols = n_cols;
    this->stride = n_cols;

    this->es = std::make_shared<float[]>(rows * cols);
}

void nn::Mat::alloc(const size_t n_rows, const size_t n_cols, const size_t n_stride)
{
    this->rows = n_rows;
    this->cols = n_cols;
    this->stride =n_stride;

    this->es = std::make_shared<float[]>(rows * cols);
}

nn::Mat nn::Mat::matRow(size_t row) const
{
    Mat m(1, cols);

    std::copy(&(this->es[row * cols]), &(this->es[(row + 1) * cols]), m.es.get());

    return m;
    
}

void nn::Mat::rand(const float low, const float max) noexcept
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) = getRandom() * (max - low) + low ;
        }
    }
}

void nn::Mat::fill(const float x) noexcept
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) = x;
        }
    }
}

void nn::Mat::dot(const Mat a, const Mat b)
{
    assert(a.cols == b.rows);
    size_t n = a.cols;
    assert(this->cols == b.cols);
    assert(this->rows == a.rows);
    
    
    for (size_t i = 0; i < this->rows; ++i)
    {
        for (size_t j = 0; j < this->cols; ++j)
        {
            for (size_t k = 0; k < n; ++k)
            {
                mat_at(this, i, j) += mat_at_non(a, i, k) * mat_at_non(b, k, j);
            }
        }        
    }    
    
}

void nn::Mat::sum(const Mat a)
{
    assert(this->rows == a.rows);
    assert(this->cols == a.cols);
    
    for (size_t i = 0; i < a.rows; ++i)
    {
        for (size_t j = 0; j < a.cols; ++j)
        {
            mat_at(this, i, j) = mat_at(this, i, j ) + mat_at_non(a, i, j);
        }        
    }
}

void nn::Mat::print(const std::string& name) const noexcept
{
    std::cout << "------------\n";
    std::cout << name << '\n';

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << mat_at(this, i, j) << " ";
        }

        std::cout << '\n';
        
    }

}

void nn::Mat::clear() noexcept
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) = 0;
    
        }
    }
}

void nn::Mat::setAt(size_t i, size_t j, float number) noexcept
{
    mat_at(this, i, j) = number;
}

float nn::Mat::getAt(const size_t i, const size_t j) const noexcept
{
    return mat_at(this, i, j);
}

void nn::Mat::apply_sigmoid() noexcept
{
    for(size_t i = 0; i < rows; ++i)
    {
        for(size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) =  sig(mat_at(this, i, j));
        }
    }
}

float nn::Mat::sig(const float x) noexcept
{
    return 1.f/(1.f + std::exp(-x));
}

float nn::Mat::getRandom() const noexcept
{
    std::random_device rd;

    std::default_random_engine engine(rd());

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}




nn::NN::NN(size_t *arch, size_t arch_count)
{
    if(arch_count < 1)
    {
        std::cout << "Arch count cannot be 0";
        return;
    }
    this->count = arch_count - 1;

    this->ws = static_cast<nn::Mat*>(malloc(sizeof(*this->ws) * this->count));
    assert(this->ws != nullptr);
        
    this->bs = static_cast<nn::Mat*>(malloc(sizeof(*this->bs) * this->count));
    assert(this->bs != nullptr);

    this->as = static_cast<nn::Mat*>(malloc(sizeof(*this->as) * (this->count + 1)));
    assert(this->as != nullptr);


    this->as[0].alloc(1, arch[0]);


    for (size_t i = 1; i < arch_count; ++i)
    {
        this->ws[i-1].alloc(this->as[i - 1].getCols(), arch[i]);
        this->bs[i-1].alloc(1, arch[i]);
        this->as[i].alloc(1, arch[i]);

    }
}

void nn::NN::print()
{
    for (size_t i = 0; i < this->count ; ++i)
    {
        std::string s = "ws[" + std::to_string(i) + "] = ";
        this->ws[i].print(s);
        s.clear();
        s = "bs[" + std::to_string(i) + "] = ";
        this->bs[i].print(s);
    }
    
}

void nn::NN::alloc(size_t *arch, size_t arch_count)
{
   if(arch_count < 1)
    {
        std::cout << "Arch count cannot be 0";
        return;
    }
    this->count = arch_count - 1;

    this->ws = static_cast<nn::Mat*>(malloc(sizeof(*this->ws) * this->count));
    assert(this->ws != nullptr);
        
    this->bs = static_cast<nn::Mat*>(malloc(sizeof(*this->bs) * this->count));
    assert(this->bs != nullptr);

    this->as = static_cast<nn::Mat*>(malloc(sizeof(*this->as) * (this->count + 1)));
    assert(this->as != nullptr);


    this->as[0].alloc(1, arch[0]);


    for (size_t i = 1; i < arch_count; ++i)
    {
        this->ws[i-1].alloc(this->as[i - 1].getCols(), arch[i]);
        this->bs[i-1].alloc(1, arch[i]);
        this->as[i].alloc(1, arch[i]);

    }
}

void nn::NN::rand(const float low, const float max)
{
    for (size_t i = 0; i < count; ++i)
    {
        this->ws[i].rand(low, max);
        this->bs[i].rand(low, max);
    }
    
}

nn::Mat nn::NN::getInput() const noexcept
{
    return this->as[0];
}

nn::Mat nn::NN::getOutput() const noexcept
{
    return this->as[count];
}

void nn::NN::setInput(const nn::Mat &m) const noexcept
{
    this->as[0] = m;
}

void nn::NN::setOutput(const nn::Mat &m) const noexcept
{
    this->as[this->count] = m;
}

size_t nn::NN::getCount() const noexcept
{
    return count;
}

void nn::NN::forward() noexcept
{
    for (size_t i = 0; i < count; ++i)
    {
        this->as[i+1].dot(this->as[i], this->ws[i]);
        this->as[i+1].sum(this->bs[i]);
        this->as[i+1].apply_sigmoid();
    }
    
}

void nn::NN::fineDiff(NN &grad, const float eps, const Mat &ti, const Mat &to)
{
    float saved;
    float c = cost(ti, to);

    for(size_t i = 0; i < count; ++i)
    {
        for(size_t j = 0; j < this->ws[i].getRows(); ++j)
        {
            for(size_t k = 0; k < this->ws[i].getCols(); ++k)
            {
                saved = this->ws[i].getAt(j, k);

                float temp = saved + eps;

                this->ws[i].setAt(j, k, temp);

                float temp2 = (this->cost(ti, to) - c)/eps;
                
                grad.setAtWs(i, j, k, temp2);
                
                this->ws[i].setAt(j, k, saved);
            }
        }

        for(size_t j = 0; j < this->bs[i].getRows(); ++j)
        {
            for(size_t k = 0; k < this->bs[i].getCols(); ++k)
            {
                saved = this->bs[i].getAt(j, k);

                float temp = saved;
                temp = temp + eps;

                this->bs[i].setAt(j, k, temp);
                float temp2 = (this->cost(ti, to) - c)/eps;
                
                grad.setAtBs(i, j, k, temp2);
                
                this->bs[i].setAt(j, k, saved);
            }
        }
    }
}

void nn::NN::learn(const NN &grad, float rate)
{
    for(size_t i = 0; i < count; ++i)
    {
        for(size_t j = 0; j < this->ws[i].getRows(); ++j)
        {
            for(size_t k = 0; k < this->ws[i].getCols(); ++k)
            {
                float temp = this->ws[i].getAt(j, k);
                float temp2 = grad.getAtWs(i, j, k);
                float temp3 = (temp2 - temp) * rate;
                
                this->ws[i].setAt(j, k, temp3);
            }
        }

        for(size_t j = 0; j < this->bs[i].getRows(); ++j)
        {
            for(size_t k = 0; k < this->bs[i].getCols(); ++k)
            {
                float temp = this->bs[i].getAt(j, k);
                float temp2 = grad.getAtBs(i, j, k);
                float temp3 = (temp2 - temp) * rate;
                this->bs[i].setAt(j, k, temp3);
            }
        }
    }
}

float nn::NN::cost(const Mat &ti, const Mat &to)
{
    assert(ti.getRows() == to.getRows());
    assert(to.getCols() == this->getOutput().getCols());

    const size_t n = ti.getRows();
    const size_t q = to.getCols();
    float c = 0.0f;

    for (size_t i = 0; i < n; ++i)
    {
        const nn::Mat x = ti.matRow(i);
        const nn::Mat y = to.matRow(i);

        this->setInput(x);

        this->forward();

        for (size_t j = 0; j < q; ++j)
        {
            float d = this->getOutput().getAt(0, j) - y.getAt(0, j);
            c = c + d*d;
        }

    }

    c = c / n;
    return c;
}

float nn::NN::getAtWs(const size_t i, const size_t j, const size_t k) const noexcept
{
    return this->ws[i].getAt(j, k);
}

void nn::NN::setAtWs(const size_t i, const size_t j, const size_t k, const float number) noexcept
{
    this->ws[i].setAt(j, k, number);
}

float nn::NN::getAtBs(const size_t i, const size_t j, const size_t k) const noexcept
{
    return this->bs[i].getAt(j, k);
}

void nn::NN::setAtBs(const size_t i, const size_t j, const size_t k, const float number) noexcept
{
    this->bs[i].setAt(j, k, number);
}

float nn::NN::getAtAs(const size_t i, const size_t j, const size_t k) const noexcept
{
    return this->as[i].getAt(j, k);
}

void nn::NN::setAtAs(const size_t i, const size_t j, const size_t k, const float number) noexcept
{
    this->as[i].setAt(j, k, number);
}