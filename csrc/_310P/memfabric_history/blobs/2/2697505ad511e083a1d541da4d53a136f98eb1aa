// SPDX-License-Identifier: Apache-2.0
// TorchNPU descriptors are mandatory: a generic from_blob alone is insufficient.
#include <torch/extension.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include <memory>

namespace py = pybind11;

namespace {

constexpr int64_t kPoolWidth = 2048;

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

void release_lease(void* ctx) {
    delete static_cast<std::shared_ptr<Lease>*>(ctx);
}

}  // namespace

at::Tensor Wrap(uint64_t ptr, int64_t rows, int64_t device, py::object owner, py::object release)
{
    TORCH_CHECK(ptr && rows > 0 && rows <= 4096, "invalid V3.2 output geometry");
    const c10::Device npu(c10::DeviceType::PrivateUse1, device);
    const int64_t nbytes = rows * kPoolWidth * static_cast<int64_t>(sizeof(at::Half));
    // The installed libtorch_npu does not export StorageDescHelper::SetDesc, so
    // storage creation goes through c10::make_storage_impl, which dispatches to
    // torch_npu's registered PrivateUse1 StorageImpl factory (NPUStorageImpl)
    // and keeps this DataPtr as-is instead of allocating.
    auto* holder = new std::shared_ptr<Lease>(std::make_shared<Lease>(std::move(owner), std::move(release)));
    auto storage = c10::make_storage_impl(
        c10::StorageImpl::use_byte_size_t{},
        nbytes,
        at::DataPtr(reinterpret_cast<void*>(ptr), holder, &release_lease, npu),
        /*allocator=*/nullptr,
        /*resizable=*/false,
        npu);
    auto* npu_storage = static_cast<torch_npu::NPUStorageImpl*>(storage.get());
    // Equivalent of StorageDescHelper::SetDesc(t, {rows, kPoolWidth}, {kPoolWidth, 1}):
    // 2-D sizes infer ACL_FORMAT_ND and InferShapeofND keeps storage sizes unchanged.
    auto& desc = npu_storage->npu_desc_;
    desc.data_type_ = caffe2::TypeMeta::Make<c10::Half>();
    desc.base_sizes_ = {rows, kPoolWidth};
    desc.base_strides_ = {kPoolWidth, 1};
    desc.storage_sizes_ = {rows, kPoolWidth};
    desc.origin_format_ = ACL_FORMAT_ND;
    desc.npu_format_ = ACL_FORMAT_ND;
    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(std::move(storage)),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        caffe2::TypeMeta::Make<c10::Half>());
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous({rows, kPoolWidth});
    return tensor;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("wrap", &Wrap); }
