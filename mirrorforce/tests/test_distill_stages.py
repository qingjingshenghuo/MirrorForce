"""恢复台账必须发现配置与产物被改写。"""
import pytest
from mirrorforce.distill.stages import StageLedger


def test_completed_stage_is_verified_not_blindly_skipped(tmp_path):
    ledger = StageLedger(tmp_path, {"seed": 42})
    artifact = tmp_path / "result"
    artifact.write_text("verified")
    ledger.mark("train", ["train", "--seed", "42"], "ok", [artifact])
    restored = StageLedger(tmp_path, {"seed": 42}, True)
    assert restored.reusable("train", ["train", "--seed", "42"])
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="产物"):
        restored.reusable("train", ["train", "--seed", "42"])


def test_resume_refuses_config_change_and_running_not_complete(tmp_path):
    ledger = StageLedger(tmp_path, {"seed": 42})
    ledger.mark("train", ["train"], "running")
    assert not ledger.reusable("train", ["train"])
    with pytest.raises(ValueError, match="配置"):
        StageLedger(tmp_path, {"seed": 43}, True)


def test_resume_refuses_external_resource_identity_change(tmp_path):
    StageLedger(tmp_path, {"seed": 42}, runtime_identity={"native": "old"})
    with pytest.raises(ValueError, match="配置"):
        StageLedger(tmp_path, {"seed": 42}, True, runtime_identity={"native": "changed"})
