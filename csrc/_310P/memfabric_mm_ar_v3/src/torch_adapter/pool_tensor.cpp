// SPDX-License-Identifier: Apache-2.0
// TorchNPU descriptors are mandatory: a generic from_blob alone is insufficient.
#include <torch/extension.h>
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include <memory>

namespace py = pybind11;
struct Lease {
    py::object owner;
    py::object release;
    Lease(py::object o, py::object r) : owner(std::move(o)), release(std::move(r)) {}
    ~Lease() {
        py::gil_scoped_acquire gil;
        try { release(); } catch (py::error_already_set& e) { e.discard_as_unraisable("V3.2 Storage release"); }
        // Drop Python references while holding the GIL, including the loaded SO.
        release = py::object();
        owner = py::object();
    }
};
at::Tensor Wrap(uint64_t ptr, int64_t rows, int64_t device, py::object owner, py::object release)
{
    TORCH_CHECK(ptr && rows > 0 && rows <= 4096, "invalid V3.2 output geometry");
    auto lease = std::make_shared<Lease>(std::move(owner), std::move(release));
    auto options = at::TensorOptions().dtype(at::kHalf).device(c10::Device(c10::DeviceType::PrivateUse1, device));
    auto tensor = at::from_blob(reinterpret_cast<void*>(ptr), {rows, 2048},
                               [lease](void*) {}, options);
    at_npu::native::StorageDescHelper::SetDesc(tensor, {rows, 2048}, {2048, 1});
    return tensor;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("wrap", &Wrap); }
