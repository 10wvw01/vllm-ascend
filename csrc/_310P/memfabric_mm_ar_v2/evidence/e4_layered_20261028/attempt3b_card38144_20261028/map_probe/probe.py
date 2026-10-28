import os, torch, torch_npu
rank = int(os.environ.get("RANK", -1))
local = int(os.environ.get("LOCAL_RANK", -1))
assert int(os.environ["WORLD_SIZE"]) == 2
torch.npu.set_device(local)
cnt = torch.npu.device_count()
props = torch.npu.get_device_properties(local)
x = torch.randn(512, 512, dtype=torch.float16, device=f"npu:{local}")
y = x @ x
torch.npu.synchronize()
s = float(y.sum().cpu())
print(f"PROBE_OK rank={rank} local={local} device_count={cnt} "
      f"props={props.name if hasattr(props,'name') else props} sum={s:.3f}", flush=True)
