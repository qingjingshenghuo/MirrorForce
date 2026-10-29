"""从当前工作区构建独立蒸馏原生模块，不覆盖原冠军部署产物。"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--runtime", type=Path, default=Path("/opt/mirrorforce"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=2)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError("输出目录已存在，不覆盖现有构建")
    stage = a.out.parent / (a.out.name + "-source")
    stage.mkdir(parents=True, exist_ok=False)
    for sub in ("duelpool", "mfenv"):
        shutil.copytree(a.repo / "mirrorforce/cxx" / sub, stage / "mirrorforce/cxx" / sub)
    source = stage / "mirrorforce/cxx/duelpool/build.py"
    spec = importlib.util.spec_from_file_location("distill_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    deps = a.runtime / "native-deps"
    manifest = stage / "mirrorforce/cxx/duelpool/deps.json"
    values = json.loads(manifest.read_text())
    values["deps"] = {name: builder.tree_digest(deps / name) for name in values["deps"]}
    values["note"] = "蒸馏精确菜单接口本地构建，使用当前工作区源码和已审计依赖"
    manifest.write_text(json.dumps(values, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(source), "--repo", str(stage), "--worktree",
        "--core", str(a.runtime / "repo/third_party/ygopro-core"), "--core-rev",
        "5760bde478561601a6ec6b4fc7f23f0ba351c3b6", "--deps", str(deps),
        "--out", str(a.out), "--jobs", str(a.jobs)], check=True)


if __name__ == "__main__":
    main()
