#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <array>

constexpr float PIF = 3.14159265358979323846264338327950288419716939937510582097494459230781640f;

namespace py = pybind11;

py::array_t<float> propagate_audio(
    float fs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rec,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& pos,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& vel,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& audio
) {

    constexpr float c = 344;

    // Validate dimensions
    if (pos.ndim() != 2 || vel.ndim() != 2 || rec.ndim() != 2 || audio.ndim() != 1) {
        throw std::invalid_argument("Expected rec,pos,vel with 2 dims and audio with 1 dim");
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

    constexpr int N = 15;
    constexpr float pin = (float)(PIF / N);
    auto sinc = [&](float x){ return x*x < 1e-12f ? 1.0f : sin(x * PIF) / (x * PIF); };
    auto impulse = [&](float x, float y){ return (0.5f + 0.5f * cos(y)) * sinc(x); };

    auto y = out.mutable_unchecked<2>();

    for(ssize_t j=0; j<m; j++){
        for(ssize_t i=0; i<n; i++){
            y(j, i) = 0.0f;
        }
    }

    for(ssize_t i=0; i<n; i++){
        for(ssize_t j=0; j<m; j++){

            float dx = p(0, i) - r(0, j);
            float dy = p(1, i) - r(1, j);
            float dz = p(2, i) - r(2, j);
            float vx = v(0, i);
            float vy = v(1, i);
            float vz = v(2, i);
            
            float absd = std::sqrt(dx*dx+dy*dy+dz*dz);

            float doppler = c / ((vx*dx + vy*dy + vz*dz) / absd + c);
            
            float sample = x(i) / (1e-9f + absd);

            double tpos = i + (double)absd / c * fs;

            int l = (int)std::ceil(tpos-N);
            int r = (int)std::floor(tpos+N);
            if(l < 0 || r >= n) continue;

            for(int ii=l; ii<=r; ii++){
                y(j, ii) += sample * impulse((ii-tpos)*doppler, (ii-tpos)*pin);
            }
        }
    }

    return out;
}


py::array_t<float> simulate_drone(
    float fs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rec,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& pos,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& vel,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& audio
) {

    constexpr float c = 344;

    // Validate dimensions
    if (pos.ndim() != 2 || vel.ndim() != 2 || rec.ndim() != 2 || audio.ndim() != 1) {
        throw std::invalid_argument("Expected rec,pos,vel with 2 dims and audio with 1 dim");
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

    constexpr int N = 15;
    constexpr float pin = (float)(PIF / N);
    auto sinc = [&](float x){ return x*x < 1e-12f ? 1.0f : sin(x * PIF) / (x * PIF); };
    auto impulse = [&](float x, float y){ return (0.5f + 0.5f * cos(y)) * sinc(x); };

    auto y = out.mutable_unchecked<2>();

    for(ssize_t j=0; j<m; j++){
        for(ssize_t i=0; i<n; i++){
            y(j, i) = 0.0f;
        }
    }

    for(ssize_t i=0; i<n; i++){
        for(ssize_t j=0; j<m; j++){

            float dx = p(0, i) - r(0, j);
            float dy = p(1, i) - r(1, j);
            float dz = p(2, i) - r(2, j);
            float vx = v(0, i);
            float vy = v(1, i);
            float vz = v(2, i);

            float theta = std::atan2(std::sqrt(dx*dx + dy*dy), dz);
            float directivity = 1.5 + 0.5 * std::cos(theta * 2);
            
            float absd = std::sqrt(dx*dx+dy*dy+dz*dz);

            float doppler = c / ((vx*dx + vy*dy + vz*dz) / absd + c);
            
            float sample = directivity * x(i) / (1e-9f + absd);

            double tpos = i + (double)absd / c * fs;

            int l = (int)std::ceil(tpos-N);
            int r = (int)std::floor(tpos+N);
            if(l < 0 || r >= n) continue;

            for(int ii=l; ii<=r; ii++){
                y(j, ii) += sample * impulse((ii-tpos)*doppler, (ii-tpos)*pin);
            }
        }
    }

    return out;
}


py::array_t<float> propagate_stationary_audio(
    float fs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rec,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& pos,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& audio
) {

    constexpr float c = 344;

    // Validate dimensions
    if (pos.ndim() != 1 || rec.ndim() != 2 || audio.ndim() != 1) {
        throw std::invalid_argument("Expected rec with 2 dims and pos,audio with 1 dim");
    }
    
    const ssize_t n = audio.shape(0);
    const ssize_t m = rec.shape(1);

    if (pos.shape(0) != 3 || rec.shape(0) != 3) {
        throw std::invalid_argument("rec must have shape (3, N), pos shape (3, 0)");
    }

    // Fast unchecked access (no bounds checks inside the loop)
    auto r = rec.unchecked<2>();
    auto p = pos.unchecked<1>();
    auto x = audio.unchecked<1>();

    py::array_t<float> out(py::buffer_info(
        nullptr,                               // no initial data: allocate
        sizeof(float),                         // itemsize
        py::format_descriptor<float>::format(),// format
        2,                                     // ndim
        { m, n },                              // shape
        { sizeof(float) * n, sizeof(float) }   // strides (bytes)
    ));

    constexpr int N = 15;
    constexpr float pin = (float)(PIF / N);
    auto sinc = [&](float x){ return x*x < 1e-12f ? 1.0f : sin(x * PIF) / (x * PIF); };
    auto impulse = [&](float x, float y){ return (0.5f + 0.5f * cos(y)) * sinc(x); };

    auto y = out.mutable_unchecked<2>();

    for(ssize_t j=0; j<m; j++){
        for(ssize_t i=0; i<n; i++){
            y(j, i) = 0.0f;
        }
    }

    constexpr int K = 2*N+1;
    std::vector<std::array<float, K> > kernel(m);
    std::vector<int> offset(m);

    for(ssize_t j=0; j<m; j++){

        float dx = p(0) - r(0, j);
        float dy = p(1) - r(1, j);
        float dz = p(2) - r(2, j);
        
        float absd = std::sqrt(dx*dx+dy*dy+dz*dz);

        double tpos = (double)absd / c * fs;

        int l = (int)std::ceil(tpos-N);
        int r = (int)std::floor(tpos+N);

        offset[j] = l;

        for(int ii=l; ii<=r; ii++){
            kernel[j][ii-l] = impulse((ii-tpos), (ii-tpos)*pin) / (1e-9f + absd);
        }
    }
    
    for(ssize_t i=0; i<n; i++){
        for(ssize_t j=0; j<m; j++){
            if(i+offset[j]+K > n || i+offset[j] < 0) continue;
            for(int k=0; k<K; k++){
                y(j, i+offset[j]+k) += x(i) * kernel[j][k];
            }
        }       
    }

    return out;
}

py::array_t<float> simulate_stationary_drone(
    float fs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rec,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& pos,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& audio
) {

    constexpr float c = 344;

    // Validate dimensions
    if (pos.ndim() != 1 || rec.ndim() != 2 || audio.ndim() != 1) {
        throw std::invalid_argument("Expected rec with 2 dims and pos,audio with 1 dim");
    }
    
    const ssize_t n = audio.shape(0);
    const ssize_t m = rec.shape(1);

    if (pos.shape(0) != 3 || rec.shape(0) != 3) {
        throw std::invalid_argument("rec must have shape (3, N), pos shape (3, 0)");
    }

    // Fast unchecked access (no bounds checks inside the loop)
    auto r = rec.unchecked<2>();
    auto p = pos.unchecked<1>();
    auto x = audio.unchecked<1>();

    py::array_t<float> out(py::buffer_info(
        nullptr,                               // no initial data: allocate
        sizeof(float),                         // itemsize
        py::format_descriptor<float>::format(),// format
        2,                                     // ndim
        { m, n },                              // shape
        { sizeof(float) * n, sizeof(float) }   // strides (bytes)
    ));

    constexpr int N = 15;
    constexpr float pin = (float)(PIF / N);
    auto sinc = [&](float x){ return x*x < 1e-12f ? 1.0f : sin(x * PIF) / (x * PIF); };
    auto impulse = [&](float x, float y){ return (0.5f + 0.5f * cos(y)) * sinc(x); };

    auto y = out.mutable_unchecked<2>();

    for(ssize_t j=0; j<m; j++){
        for(ssize_t i=0; i<n; i++){
            y(j, i) = 0.0f;
        }
    }

    constexpr int K = 2*N+1;
    std::vector<std::array<float, K> > kernel(m);
    std::vector<int> offset(m);

    for(ssize_t j=0; j<m; j++){

        float dx = p(0) - r(0, j);
        float dy = p(1) - r(1, j);
        float dz = p(2) - r(2, j);
        
        float absd = std::sqrt(dx*dx+dy*dy+dz*dz);

        float theta = std::atan2(std::sqrt(dx*dx + dy*dy), dz);
        float directivity = 1.5 + 0.5 * std::cos(theta * 2);

        double tpos = (double)absd / c * fs;

        int l = (int)std::ceil(tpos-N);
        int r = (int)std::floor(tpos+N);

        offset[j] = l;

        for(int ii=l; ii<=r; ii++){
            kernel[j][ii-l] = directivity * impulse((ii-tpos), (ii-tpos)*pin) / (1e-9f + absd);
        }
    }
    
    for(ssize_t i=0; i<n; i++){
        for(ssize_t j=0; j<m; j++){
            if(i+offset[j]+K > n || i+offset[j] < 0) continue;
            for(int k=0; k<K; k++){
                y(j, i+offset[j]+k) += x(i) * kernel[j][k];
            }
        }       
    }

    return out;
}

py::array_t<float> propagate_planewave(
    float fs,
    float dist,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rec,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& dir,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& audio
) {

    constexpr float c = 344;

    // Validate dimensions
    if (dir.ndim() != 2 || rec.ndim() != 2 || audio.ndim() != 1) {
        throw std::invalid_argument("Expected rec,pos,vel with 2 dims and audio with 1 dim");
    }
    
    const ssize_t n = audio.shape(0);
    const ssize_t m = rec.shape(1);

    if (dir.shape(0) != 3 || rec.shape(0) != 3) {
        throw std::invalid_argument("rec,pos,vel must have shape (3, N)");
    }

    if (dir.shape(1) != n) {
        throw std::invalid_argument("pos,vel must have shape (3, N) when audio has shape (N)");
    }

    // Fast unchecked access (no bounds checks inside the loop)
    auto r = rec.unchecked<2>();
    auto p = dir.unchecked<2>();
    auto x = audio.unchecked<1>();

    py::array_t<float> out(py::buffer_info(
        nullptr,                               // no initial data: allocate
        sizeof(float),                         // itemsize
        py::format_descriptor<float>::format(),// format
        2,                                     // ndim
        { m, n },                              // shape
        { sizeof(float) * n, sizeof(float) }   // strides (bytes)
    ));

    constexpr int N = 15;
    constexpr float pin = (float)(PIF / N);
    auto sinc = [&](float x){ return x*x < 1e-12f ? 1.0f : sin(x * PIF) / (x * PIF); };
    auto impulse = [&](float x, float y){ return (0.5f + 0.5f * cos(y)) * sinc(x); };

    auto y = out.mutable_unchecked<2>();

    for(ssize_t j=0; j<m; j++){
        for(ssize_t i=0; i<n; i++){
            y(j, i) = 0.0f;
        }
    }

    for(ssize_t i=0; i<n; i++){

        for(ssize_t j=0; j<m; j++){

            float dx = p(0, i) * r(0, j);
            float dy = p(1, i) * r(1, j);
            float dz = p(2, i) * r(2, j);

            float absd = dist - (dx + dy + dz);

            double tpos = i + (double)absd / c * fs;

            int l = (int)std::ceil(tpos-N);
            int r = (int)std::floor(tpos+N);
            if(l < 0 || r >= n) continue;

            for(int ii=l; ii<=r; ii++){
                y(j, ii) += x(i) * impulse((ii-tpos), (ii-tpos)*pin);
            }
        }
    }

    return out;
}

py::array_t<float> propagate_stationary_planewave(
    float fs,
    float dist,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rec,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& dir,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& audio
) {

    constexpr float c = 344;

    // Validate dimensions
    if (dir.ndim() != 1 || rec.ndim() != 2 || audio.ndim() != 1) {
        throw std::invalid_argument("Expected rec with 2 dims and dir,audio with 1 dim");
    }
    
    const ssize_t n = audio.shape(0);
    const ssize_t m = rec.shape(1);

    if (dir.shape(0) != 3 || rec.shape(0) != 3) {
        throw std::invalid_argument("rec must have shape (3, N), dir shape (3, 0)");
    }

    // Fast unchecked access (no bounds checks inside the loop)
    auto r = rec.unchecked<2>();
    auto p = dir.unchecked<1>();
    auto x = audio.unchecked<1>();

    py::array_t<float> out(py::buffer_info(
        nullptr,                               // no initial data: allocate
        sizeof(float),                         // itemsize
        py::format_descriptor<float>::format(),// format
        2,                                     // ndim
        { m, n },                              // shape
        { sizeof(float) * n, sizeof(float) }   // strides (bytes)
    ));

    constexpr int N = 15;
    constexpr float pin = (float)(PIF / N);
    auto sinc = [&](float x){ return x*x < 1e-12f ? 1.0f : sin(x * PIF) / (x * PIF); };
    auto impulse = [&](float x, float y){ return (0.5f + 0.5f * cos(y)) * sinc(x); };

    auto y = out.mutable_unchecked<2>();

    for(ssize_t j=0; j<m; j++){
        for(ssize_t i=0; i<n; i++){
            y(j, i) = 0.0f;
        }
    }

    constexpr int K = 2*N+1;
    std::vector<std::array<float, K> > kernel(m);
    std::vector<int> offset(m);

    for(ssize_t j=0; j<m; j++){

        float dx = p(0) * r(0, j);
        float dy = p(1) * r(1, j);
        float dz = p(2) * r(2, j);

        float absd = dist - (dx + dy + dz);

        double tpos = (double)absd / c * fs;

        int l = (int)std::ceil(tpos-N);
        int r = (int)std::floor(tpos+N);

        offset[j] = l;

        for(int ii=l; ii<=r; ii++){
            kernel[j][ii-l] = impulse((ii-tpos), (ii-tpos)*pin);
        }
    }
    
    for(ssize_t i=0; i<n; i++){
        for(ssize_t j=0; j<m; j++){
            if(i+offset[j]+K > n || i+offset[j] < 0) continue;
            for(int k=0; k<K; k++){
                y(j, i+offset[j]+k) += x(i) * kernel[j][k];
            }
        }       
    }

    return out;
}

PYBIND11_MODULE(propagationmod, m) {
    m.doc() = "Cpp module";

    m.def("simulate_audio_propagation", &propagate_audio,
          py::arg("fs"), py::arg("rec"), py::arg("pos"), py::arg("vel"), py::arg("audio"),
          R"doc(audio propagation)doc");
    
    m.def("simulate_drone", &simulate_drone,
          py::arg("fs"), py::arg("rec"), py::arg("pos"), py::arg("vel"), py::arg("audio"),
          R"doc(drone audio propagation)doc");

    m.def("static_audio_propagation", &propagate_stationary_audio,
          py::arg("fs"), py::arg("rec"), py::arg("pos"), py::arg("audio"),
          R"doc(audio propagation from static point)doc");

    m.def("simulate_static_drone", &simulate_stationary_drone,
          py::arg("fs"), py::arg("rec"), py::arg("pos"), py::arg("audio"),
          R"doc(stationary drone audio propagation)doc");
    
    m.def("planewave_propagation", &propagate_planewave,
          py::arg("fs"), py::arg("dist"), py::arg("rec"), py::arg("dir"), py::arg("audio"),
          R"doc(plane wave propagation. dir assumed unitary)doc");

    m.def("static_planewave_propagation", &propagate_stationary_planewave,
          py::arg("fs"), py::arg("dist"), py::arg("rec"), py::arg("dir"), py::arg("audio"),
          R"doc(plane wave propagation. dir assumed unitary)doc");
}
