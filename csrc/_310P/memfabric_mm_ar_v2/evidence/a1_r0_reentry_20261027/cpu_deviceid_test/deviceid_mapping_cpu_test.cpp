// Pure-CPU boundary test replicating the EXACT HybmPhysicalDeviceIdOf logic
// from /tmp/opencode/mf-fix/src/hybm/csrc/common/hybm_define.h @ commit 5802f550
// (historical fix validated on-device 2026-10-23). No NPU, no CANN, no SO.
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

static uint32_t HybmPhysicalDeviceIdOf(uint32_t logicDevId)
{
    const char *vis = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    if (vis == nullptr || *vis == '\0') {
        return logicDevId;
    }
    uint32_t idx = 0;
    const char *p = vis;
    while (*p != '\0') {
        char *end = nullptr;
        const unsigned long v = std::strtoul(p, &end, 10);
        if (end == p) {
            break;
        }
        if (idx == logicDevId) {
            return static_cast<uint32_t>(v);
        }
        ++idx;
        p = (*end == ',') ? end + 1 : end;
        while (*p == ' ') {
            ++p;
        }
    }
    return logicDevId;
}

static int failures = 0;
static void expect(const char* vis, uint32_t logic, uint32_t want) {
    if (vis) { char buf[64]; snprintf(buf, sizeof buf, "ASCEND_RT_VISIBLE_DEVICES=%s", vis); putenv(strdup(buf)); }
    else { unsetenv("ASCEND_RT_VISIBLE_DEVICES"); }
    uint32_t got = HybmPhysicalDeviceIdOf(logic);
    printf("vis=%-22s logic=%u -> physical=%u (expect %u) %s\n",
           vis ? vis : "<unset>", logic, got, want, got == want ? "PASS" : "FAIL");
    if (got != want) ++failures;
}

int main() {
    expect(nullptr, 0, 0);            // unset: identity
    expect(nullptr, 1, 1);
    expect("", 3, 3);                 // empty: identity
    expect("0,1", 0, 0);              // identity mapping
    expect("0,1", 1, 1);
    expect("2,3", 0, 2);              // non-zero origin (card 16384 dies)
    expect("2,3", 1, 3);
    expect("6,7", 0, 6);              // historical on-device case (card 43520)
    expect("6,7", 1, 7);
    expect("4,5", 0, 4);              // card 38144 dies
    expect("4,5", 1, 5);
    expect(" 2 , 3 ", 0, 2);          // tolerated spaces
    expect("x,y", 0, 0);               // malformed: identity fallback
    expect("2", 1, 1);                // short list: identity fallback
    expect("2,3", 7, 7);              // logic beyond list: identity fallback
    printf(failures ? "RESULT=FAIL(%d)\n" : "RESULT=PASS(%d failures)\n", failures);
    return failures ? 1 : 0;
}
