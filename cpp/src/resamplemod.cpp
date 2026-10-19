#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

constexpr float PIF = 3.14159265358979323846264338327950288419716939937510582097494459230781640f;

namespace py = pybind11;

py::array_t<float> sinc_resample(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& IN,
    double ratio,
    int zeros)
{
   
    int n = IN.shape(0);
    int m = std::ceil(n * ratio);
    
    py::array_t<float> OUT(py::buffer_info(
        nullptr,                               // no initial data: allocate
        sizeof(float),                         // itemsize
        py::format_descriptor<float>::format(),// format
        1,                                     // ndim
        { m },                                 // shape
        { sizeof(float) }                      // strides (bytes)
    ));
    auto out = OUT.mutable_unchecked<1>();
    auto in = IN.unchecked<1>();

    double iratio = 1.0 / ratio;
    double dzeros = zeros;

    auto sinc = [&](float x) -> float {
        if(x*x < 1e-12f) return 1.0f;
        return std::sin(PIF * x) / (PIF * x);
    };
    
    auto window = [&](float x) -> float {
        float y = 0.5f + 0.5f * std::cos(PIF * x / zeros);
        return y;
    };

    for(ssize_t j=0; j<m; j++) out(j) = 0.0f;

    // for(int i=0; i<n; i++){
        
    //     int l = std::floor(i * ratio), r = std::ceil(i * ratio);
    //     if(l == r) r++;

    //     l = std::min(m-1, l);
    //     double ll = l*iratio, rr = r*iratio;

    //     for(;l >= 0 && i - ll <= dzeros; l--, ll-=iratio) out(l) += in(i)*sinc(i-ll)*window(i-ll);
    //     for(;r < m && rr - i <= dzeros; r++, rr+=iratio) out(r) += in(i)*sinc(rr-i)*window(rr-i);
    // }

    for(int i=0; i<n; i++){

        int l = (int)std::ceil((i-dzeros) * ratio);
        int r = (int)std::floor((i+dzeros) * ratio);
        l = std::max(0, l);
        r = std::min(r, m-1);

        for(int ii=l; ii<=r; ii++){
            double d = i - ii*iratio;
            out(ii) += in(i)*sinc(d)*window(d);
        }
    }

    return OUT;
}

PYBIND11_MODULE(resamplemod, m) {
    m.doc() = "Cpp resample";

    m.def("sinc_resample", &sinc_resample,
          py::arg("IN"), py::arg("ratio"), py::arg("zeros"),
          R"doc(audio resampling)doc");
}
