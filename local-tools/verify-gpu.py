"""验证实际 GPU 计算，拒绝静默回退 CPU；不读取或修改模型参数。"""
from pathlib import Path
import json

import jax
import jax.numpy as jnp


assert jax.__version__ == '0.5.3', 'JAX 版本与项目不一致'
devices = jax.devices()
assert devices and all(device.platform == 'gpu' for device in devices), devices
result = jax.jit(lambda x: x @ x)(jnp.ones((64, 64), dtype=jnp.bfloat16))
total = float(jnp.sum(result, dtype=jnp.float32).block_until_ready())
assert total == 262144.0, total
record = {'jax': jax.__version__, 'backend': jax.default_backend(),
          'devices': [str(device) for device in devices],
          'device_kind': [device.device_kind for device in devices],
          'bfloat16_matmul_sum': total, 'passed': True}
Path('/opt/mirrorforce/logs/gpu-check.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, ensure_ascii=False), flush=True)
