#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

constexpr float PIF = 3.14159265358979323846264338327950288419716939937510582097494459230781640f;

namespace py = pybind11;

py::array_t<float> propagate_audio(
    float fs,
    py::array_t<float, py::array::c_style | py::array::forcecast> rec,
    py::array_t<float, py::array::c_style | py::array::forcecast> pos,
    py::array_t<float, py::array::c_style | py::array::forcecast> vel,
    py::array_t<float, py::array::c_style | py::array::forcecast> audio
) {

    constexpr float c = 344;

    // Validate dimensions
    if (pos.ndim() != 2 || vel.ndim() != 2 || rec.ndim() != 2 || audio.ndim() != 1) {
        throw std::invalid_argument("Expected rec,pos,vel with 2 dims and w with 1 dim");
    }
    
    const ssize_t n = audio.shape(0);
    const ssize_t m = rec.shape(1);

    if (pos.shape(0) != 3 || vel.shape(0) != 3 || rec.shape(0) != 3) {
        throw std::invalid_argument("rec,pos,vel must have shape (3, N)");
    }

    if (pos.shape(1) != n || vel.shape(1) != n) {
        throw std::invalid_argument("pos,vel must have shape (3, N) when audio has shape (N)");
    }

    // Fast unchecked access (no bounds checks inside the loop)
    auto r = rec.unchecked<2>();
    auto p = pos.unchecked<2>();
    auto v = vel.unchecked<2>();
    auto x = audio.unchecked<1>();

    py::array_t<float> out(py::buffer_info(
        nullptr,                               // no initial data: allocate
        sizeof(float),                         // itemsize
        py::format_descriptor<float>::format(),// format
        2,                                     // ndim
        { m, n },                              // shape
        { sizeof(float) * n, sizeof(float) }   // strides (bytes)
    ));

    constexpr int N = 12;
    constexpr float pin = (float)(PIF / N);
    auto sinc = [&](float x){ return x*x < 1e-12f ? 1.0f : sin(x * PIF) / (x * PIF); };
    auto impulse = [&](float x, float y){ return (0.5f + 0.5f * cos(y)) * sinc(x); };
    auto write_sample = [&](float *x, float p, float s, float d){
        int l = (int)std::ceil(p-N);
        int r = (int)std::floor(p+N);
        for(int i=l; i<=r; i++) x[i] += s * impulse((i-p)*d, (i-p)*pin);
    };

    auto y = out.mutable_unchecked<2>();

    for(ssize_t j=0; j<m; j++){
        for(ssize_t i=0; i<n; i++){
            y(j, i) = 0.0f;
        }
    }

    for(ssize_t i=0; i<n; i++){
        for(ssize_t j=0; j<m; j++){
            y(j, i) = x(i);

            float dx = p(i, 0) - r(j, 0);
            float dy = p(i, 1) - r(j, 1);
            float dz = p(i, 2) - r(j, 2);
            float vx = v(i, 0);
            float vy = v(i, 1);
            float vz = v(i, 2);
            
            float absd = std::sqrt(dx*dx+dy*dy+dz*dz);

            float doppler = c / ((vx*dx + vy*dy + vz*dz) / absd + c);
            
            float sample = x(i) / (1.0f + absd);

            float tpos = absd / c * fs;

            int l = (int)std::ceil(tpos-N);
            int r = (int)std::floor(tpos+N);
            for(int i=l; i<=r; i++){
                y(j, i) += doppler * impulse((i-tpos)*doppler, (i-tpos)*pin);
            }
        }
    }

    return out;
}

PYBIND11_MODULE(propagationmod, m) {
    m.doc() = "Example module: process 3xN arrays and weights to produce N float outputs";

    m.def("process_arrays", &propagate_audio,
          py::arg("fs"), py::arg("rec"), py::arg("pos"), py::arg("vel"), py::arg("audio"),
          R"doc(
Process two 3xN arrays A, B and a length-N weight vector w to produce an N-length float array.

Arguments:
  A: np.ndarray, shape (3, N), float or double
  B: np.ndarray, shape (3, N), float or double
  w: np.ndarray, shape (N,), float32 preferred (will be cast if needed)

Returns:
  np.ndarray, shape (N,), dtype float32

Computation (example):
  out[n] = dot(A[:, n], B[:, n]) * w[n]
)doc");
}
