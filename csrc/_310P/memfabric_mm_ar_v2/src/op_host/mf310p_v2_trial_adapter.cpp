// 正式 T0 Host 运行时：建池、IB 自检、融合发射、协议状态和销毁。
#include "mf310p_v2_trial_internal.h"
#include <cstdlib>
#include <new>

extern "C" int mf310p_v2_trial_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_ib_probe_async(
    uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_ib_fan_probe_async(
    uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);
extern "C" uint32_t mf310p_v2_trial_ib_workspace_bytes();

namespace {
/** @brief 统一释放 Host 负责的 IB、状态、自检缓冲和 V1 公共对称池。
 * @param state 持有所有资源的上下文，须非空。@return 无，供创建失败回滚和正常销毁复用。
 * @why 防止跨多步建池失败时漏释放已成功分配的设备内存。
 */
void free_allocations(mf310p_v2_trial* state)
{
    if (state->ib) aclrtFree(state->ib);
    if (state->status) aclrtFree(state->status);
    if (state->probe_result) aclrtFree(state->probe_result);
    if (state->v1_pool) mf310p_destroy(state->v1_pool);
}
}
/** @brief 建立两 Rank 对称内存会话，获取对端几何并创建正式 Kernel 的工作区。
 * @param rank 本地 TP rank(0/1)；store_url 双 Rank 同一 rendezvous 地址。
 * @param chunk_rows 静态块行数 256/512/1024；out 输出句柄（失败置空）。
 * @return 0 成功，-1 参数错误，-2 Host 分配失败，或底层 ACL/MemFabric 非零错误。
 * @why 先同步 geometry，再读取 layout；设备端依赖交换后的 peer GVA 和控制区地址。
 */
extern "C" int mf310p_v2_trial_create(
    int rank, const char* store_url, uint32_t chunk_rows, mf310p_v2_trial_t** out)
{
    if (!out || !store_url || (rank != 0 && rank != 1) ||
        (chunk_rows != 256 && chunk_rows != 512 && chunk_rows != 1024))
        return -1;
    *out = nullptr;
    auto* s = new (std::nothrow) mf310p_v2_trial();
    if (!s) return -2;
    s->rank = static_cast<uint32_t>(rank);
    s->chunk_rows = chunk_rows;
    // 先复用 V1 的公共 MemFabric 建池与 barrier/exchange API；共享的是运输 ABI，
    // 不是运行 V1 的 MM Kernel。必须在 exchange 后读取布局，防止持有旧对端地址。
    int err = mf310p_create(rank, 2, store_url, kPoolSize, kWindowRows,
                           chunk_rows, &s->v1_pool);
    if (!err) err = mf310p_control_barrier(s->v1_pool);
    if (!err) err = mf310p_exchange_geometry(s->v1_pool);
    /* exchange_geometry 完成之后才可以读取最新布局；expected_recv_base 和 expected_credit_dst 是交换过程中填入的对端 GVA，遵循 V1 交换后的布局刷新纪律。 */
    if (!err) err = mf310p_get_layout(s->v1_pool, &s->layout);
    // IB workspace 大小取设备侧导出值；status/probe_result 独立分配并初始化为零，
    // 保持核间同步事件和故障码与对称数据 C0/C1 分区隔离。
    if (!err) {
        err = static_cast<int>(aclrtMalloc(
            &s->ib, mf310p_v2_trial_ib_workspace_bytes(),
            ACL_MEM_MALLOC_HUGE_FIRST));
    }
    if (!err) err = static_cast<int>(aclrtMalloc(
        &s->status, kStatusSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!err) err = static_cast<int>(aclrtMalloc(
        &s->probe_result, kProbeResultBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!err) err = static_cast<int>(aclrtMemset(
        s->ib, mf310p_v2_trial_ib_workspace_bytes(), 0,
        mf310p_v2_trial_ib_workspace_bytes()));
    if (!err) err = static_cast<int>(aclrtMemset(
        s->status, kStatusSize, 0, kStatusSize));
    if (!err) err = static_cast<int>(aclrtMemset(
        s->probe_result, kProbeResultBytes, 0, kProbeResultBytes));
    if (!err) err = static_cast<int>(aclrtMemset(
        reinterpret_cast<void*>(s->layout.ack_slot), 8, 0, 8));
    if (err) {
        free_allocations(s);
        delete s;
        return err;
    }
    *out = s;
    return 0;
}

/** @brief 在正式 run 之前检查官方 IB 的双核循环与 8 核 fan-in/fan-out 协议。
 * @param s V2 context；repetitions 往复轮数；stream ACL stream 指针。
 * @return 0 并设置 ib_probed；-1 入参错误，-2/-3 事件错误，或 ACL 非零错误。
 * @why 正式 Kernel 依赖这些事件收敛和放行；此门禁虽叫 probe，但 Serving 必须经过。
 */
extern "C" int mf310p_v2_trial_ib_probe(
    mf310p_v2_trial_t* s, uint32_t repetitions, void* stream)
{
    if (!s || !stream || repetitions == 0 || repetitions > 100000)
        return -1;
    aclrtStream acl_stream = static_cast<aclrtStream>(stream);
    int err = mf310p_v2_ib_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result), 0, 0, acl_stream);
    if (!err) err = mf310p_v2_ib_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result), 0, 0, acl_stream);
    if (!err) err = mf310p_v2_ib_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result),
        repetitions, 1, acl_stream);
    if (!err) err = static_cast<int>(aclrtSynchronizeStream(acl_stream));
    uint32_t slots[8 * 16] = {};  /* 8 个核各占 64B，即每核 16 个 uint32。 */
    if (!err) err = static_cast<int>(aclrtMemcpy(
        slots, sizeof(slots), s->probe_result, sizeof(slots),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (!err && (slots[0] != repetitions || slots[16] != repetitions))
        return -2;
    // 真实 MM Kernel 执行 8→1 就绪、1→7 释放、7→1 完成会合。
    // 启用前需单独对该设备符号进行两次预热。
    if (!err) err = mf310p_v2_ib_fan_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result), 0, 0, acl_stream);
    if (!err) err = mf310p_v2_ib_fan_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result), 0, 0, acl_stream);
    if (!err) err = mf310p_v2_ib_fan_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result),
        repetitions, 1, acl_stream);
    if (!err) err = static_cast<int>(aclrtSynchronizeStream(acl_stream));
    if (!err) err = static_cast<int>(aclrtMemcpy(
        slots, sizeof(slots), s->probe_result, sizeof(slots),
        ACL_MEMCPY_DEVICE_TO_HOST));
    bool all_ok = err == 0;
    for (uint32_t c = 0; c < 8; ++c) {
        all_ok = all_ok && slots[16 * c] == repetitions;
    }
    s->ib_probed = all_ok;
    return err ? err : (all_ok ? 0 : -3);
}

/** @brief 把一个完整 chunk 序列的本地 MM+SDMA+Add wave 异步入队。
 * @param s 已创建且完成 IB 自检的上下文；x/w/y 分别是设备输入、NZ 权重、ND 输出指针。
 * @param m 行数（chunk 整倍数且≤8192）；overlap 0=串行/1=流水；stream 执行流。
 * @return 0=发射成功，失败返回非零；不意味着 Kernel 已完成，应在同步后检查 status。
 * @why 集中管理 Rank0/Rank1 两侧 send/recv 的地址重映射、预热和对端邮件地址期望值。
 */
extern "C" int mf310p_v2_trial_run(
    mf310p_v2_trial_t* s, uint64_t x, uint64_t w, uint64_t y,
    uint32_t m, uint32_t overlap, void* stream)
{
    if (!s || !s->ib_probed || !stream || !x || !w || !y ||
        !m || m > kWindowRows || m % s->chunk_rows || overlap > 1) {
        return -1;
    }
    auto acl_stream = static_cast<aclrtStream>(stream);
    const auto& l = s->layout;
    const bool rank0 = s->rank == 0;
    // 逻辑对称输出：C0 从偏移 0 开始，C1 从 arena_bytes 开始。
    // 每个 Rank 只向自己的 C 区做 MM，向另一 Rank 的同名逻辑区发 SDMA；
    // Rank1 的 own/recv 对应布局与 Rank0 互换，不能简单对两 Rank 复用相同偏移表达式。
    const uint64_t ownC = rank0 ? l.send_arena : l.recv_arena;
    const uint64_t peerC = rank0 ? l.recv_arena : l.send_arena;
    const uint64_t peerDst = rank0 ? l.peer_segment : l.peer_recv_arena;
    const uint64_t expectedRecvDst =
        rank0 ? l.expected_recv_base : l.expected_recv_base - l.arena_bytes;
    /* T0 BUG#3 首写保护及 2026-10-17 实测补充：建池后新分配缓冲区的首次 MTE3 输出可能静默丢失；只有 aclrtSynchronizeDevice 能缓解，Stream 同步和 enable=0 预热无效。此外，V2TrialKernel 的首次真实发射即使写入长期预热的 y 也可能丢失结果，因此每个符号首次真实执行也需设备同步。建池后全设备同步可能耗时 0～15 秒，双 Rank 偏斜曾引发 AICore 等待超时。T1 小 M 路径使用开销较小的 Stream 级守护。 */
    const bool first_real_symbol_launch = !s->trial_symbol_launched;
    // 已归档的首次真实发射/MTE3 首写问题工程规避：先执行两次 enable=0 的设备符号预热。
    // 这不是 Kernel 正确性本身的证明，也不会避免所有冷启动异常。
    if (!s->trial_symbol_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int rc = mf310p_v2_trial_launch_async(
                l.pool_base, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                s->chunk_rows, 1, 0, 0, acl_stream);
            if (rc != 0) return rc;
        }
        s->trial_symbol_warmed = true;
    }
    // 有界记录曾完成首写保护的输出地址，避免每层重复触发高代价设备同步。
    // 地址可能被释放后重用，调用方须保证标记地址对应的 Tensor 生命周期持续有效。
    bool output_warmed = false;
    for (uint64_t known : s->trial_warmed_ys) {
        if (known == y) {
            output_warmed = true;
            break;
        }
    }
    if (getenv("MF310P_V2_NOWARMUP") == nullptr &&
        (first_real_symbol_launch || !output_warmed)) {
        const int sync_err = static_cast<int>(aclrtSynchronizeDevice());
        if (sync_err) return sync_err;
        // 服务为该上下文的整个生命周期固定输出分配。
        // 试验调用方也必须保证已标记为预热的地址保持存活，
        // 释放后重新分配到相同地址的情况不能
        // 由该地址缓存识别。
        if (!output_warmed) {
            s->trial_warmed_ys[s->trial_warmed_y_next] = y;
            s->trial_warmed_y_next =
                (s->trial_warmed_y_next + 1) % 64;
        }
    }
    s->warmed_y = y;
    s->trial_symbol_launched = true;
    // 每波入队前在同一个 stream 上清零错误槽，再发 Kernel；否则可能读取上一波状态。
    // 返回后仍为异步执行，不能在此误称 status=0 即成功。
    const int clear_err = static_cast<int>(aclrtMemsetAsync(
        s->status, kStatusSize, 0, kStatusSize, acl_stream));
    if (clear_err) return clear_err;
    return mf310p_v2_trial_launch_async(
        l.pool_base, x, w, y, ownC, peerC, peerDst, expectedRecvDst,
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->status),
        l.ack_slot, l.peer_ack_slot, l.expected_credit_dst,
        s->chunk_rows, m / s->chunk_rows, overlap, 1, acl_stream);
}


/** @brief 复制 Kernel 写入的 64 位协议错误码到 Host。
 * @param s 已建立的上下文；status 输出错误码指针（0 表示当前未记录设备协议失败）。
 * @return ACL memcpy 返回值，-1 为非法指针；调用前应明确同步对应执行流。
 */
extern "C" int mf310p_v2_trial_check_status(
    mf310p_v2_trial_t* s, uint64_t* status)
{
    if (!s || !status) return -1;
    return static_cast<int>(aclrtMemcpy(
        status, sizeof(*status), s->status, sizeof(*status),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

/** @brief 释放 V2 context 所有资源并销毁不透明句柄。
 * @param s 上下文，可空。@return 0；要求在途 Kernel/SDMA 已经安全终止。
 */
extern "C" int mf310p_v2_trial_destroy(mf310p_v2_trial_t* s)
{
    if (!s) return 0;
    free_allocations(s);
    delete s;
    return 0;
}
