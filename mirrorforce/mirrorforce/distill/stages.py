"""可恢复阶段台账；已完成任务绑定命令与产物内容。"""
import hashlib
import json
import subprocess
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity():
    """同时记录 HEAD 和当前源码内容，未提交改动不能冒充原提交。"""
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root).split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(set(n for n in names if n)):
        path = root / name.decode("utf-8")
        if not path.is_file() or path.suffix.lower() not in (".py", ".cc", ".cpp", ".c", ".h", ".hpp", ".json", ".txt", ".sh", ".ps1", ".cmd", ".toml", ".yaml", ".yml"):
            continue
        digest.update(name + b"\0" + file_hash(path).encode())
    return {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_tree_sha256": digest.hexdigest()}


class StageLedger:
    def __init__(self, root, config, resume=False, runtime_identity=None):
        self.root = Path(root)
        self.path = self.root / "stages.json"
        self.config_sha = hashlib.sha256(json.dumps({"config": config, "runtime": runtime_identity or {}}, sort_keys=True).encode()).hexdigest()
        if resume:
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state["config_sha"] != self.config_sha:
                raise ValueError("恢复配置与原实验不一致")
        else:
            if self.path.exists():
                raise ValueError("阶段台账已存在，请显式恢复")
            self.state = {"config_sha": self.config_sha, "stages": {}}
            self.flush()

    def flush(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.path)

    def reusable(self, name, command):
        record = self.state["stages"].get(name)
        if not record or record["status"] != "ok":
            return False
        if record["command"] != command:
            raise ValueError("已完成阶段的命令变化：" + name)
        for path, expected in record["artifacts"].items():
            if not Path(path).is_file() or file_hash(path) != expected:
                raise ValueError("已完成阶段的产物缺失或变化：" + path)
        return True

    def mark(self, name, command, status, artifacts=()):
        self.state["stages"][name] = {"command": command, "status": status,
            "artifacts": {str(Path(p).resolve()): file_hash(p) for p in artifacts}}
        self.flush()
