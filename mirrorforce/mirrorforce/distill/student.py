"""标准因果语言模型动作 token、受约束生成及新增词表行微调。"""
import json
from pathlib import Path
from .data import TOKENS, prompt_text

FORMAT = "standard_causal_lm_actions/v2"


def action_count(row):
    actions = row.get("legal_actions", row.get("view", {}).get("actions"))
    if not actions or not 1 <= len(actions) <= 192:
        raise ValueError("合法动作数必须在 1～192")
    return len(actions)


class Student:
    def __init__(self, base="Qwen/Qwen2.5-1.5B-Instruct", adapter=None, device="cuda", training=False,
                 revision=None, seed=42, context_limit=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, PeftModel
        from safetensors.torch import load_file
        torch.manual_seed(seed)
        self.torch, self.device, self.training = torch, device, training
        saved = json.loads((Path(adapter) / "student.json").read_text()) if adapter else {}
        if saved and saved.get("format") != FORMAT:
            raise ValueError("旧动作头检查点不能作为标准 LM 检查点恢复")
        self.base, self.revision = saved.get("base", base), saved.get("revision", revision)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or self.base, revision=None if adapter else self.revision)
        self.base_tokenizer_size = saved.get("base_tokenizer_size", len(self.tokenizer))
        if not adapter and any(token in self.tokenizer.get_vocab() for token in TOKENS):
            raise ValueError("基础 tokenizer 已包含动作标记，需要明确的检查点身份")
        self.tokenizer.add_special_tokens({"additional_special_tokens": TOKENS})
        self.ids = self.tokenizer.convert_tokens_to_ids(TOKENS)
        if len(set(self.ids)) != 192 or any(self.tokenizer.encode(t, add_special_tokens=False) != [i] for t, i in zip(TOKENS, self.ids)):
            raise ValueError("动作标记必须是不同的单 token")
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.base, revision=self.revision,
            torch_dtype=dtype, low_cpu_mem_usage=True)
        self.revision = getattr(model.config, "_commit_hash", None) or self.revision
        maximum = int(model.config.max_position_embeddings)
        self.context_limit = (int(context_limit) if context_limit is not None else
                              min(int(saved.get("context_limit", 4096)), maximum))
        if not 1 < self.context_limit <= maximum:
            raise ValueError("上下文上限超出模型位置编码范围")
        self.original_vocab = model.get_input_embeddings().num_embeddings
        if min(self.ids) < self.base_tokenizer_size:
            raise ValueError("动作标记与基础词表碰撞")
        # Qwen权重有补齐的词表行，tokenizer扩展可能落在原来未分配的行中。
        # 不得把这些有效新token误判为已有词碰撞，也不能缩掉原模型的补齐行。
        model.resize_token_embeddings(max(self.original_vocab, len(self.tokenizer)), mean_resizing=False)
        model.to(device)
        self.tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
        with torch.no_grad():
            for parameter in self._embedding_parameters(model):
                mean = parameter[:self.base_tokenizer_size].float().mean(0).to(parameter.dtype)
                parameter[self.ids] = mean + torch.randn_like(parameter[self.ids]) * .001
            if adapter:
                tensors = load_file(str(Path(adapter) / "action_rows.safetensors"), device=str(device))
                if saved["action_token_ids"] != self.ids or saved["tied"] != self.tied:
                    raise ValueError("词表映射或共享关系不一致")
                model.get_input_embeddings().weight[self.ids] = tensors["input_rows"]
                if not self.tied:
                    model.get_output_embeddings().weight[self.ids] = tensors["output_rows"]
        self.model = (PeftModel.from_pretrained(model, adapter, is_trainable=training) if adapter else
            get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")))
        self.row_parameters = self._embedding_parameters(self.model)
        self._hooks = []
        for parameter in self.row_parameters:
            parameter.requires_grad_(training)
            if training:
                self._hooks.append(parameter.register_hook(self._row_gradient))
        self.model.train(training)
        if training:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

    @staticmethod
    def _embedding_parameters(model):
        values = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
        return list({id(p): p for p in values}.values())

    def _row_gradient(self, grad):
        # 此组 decay 必须为零，确保旧词表行及其 Adam 状态不变。
        masked = self.torch.zeros_like(grad)
        masked[self.ids] = grad[self.ids]
        return masked

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def optimizer_groups(self, lr=1e-4):
        rows = {id(p) for p in self.row_parameters}
        return [{"params": [p for p in self.parameters() if id(p) not in rows], "lr": lr, "weight_decay": .01},
                {"params": self.row_parameters, "lr": lr, "weight_decay": 0.}]

    def _inputs(self, rows):
        torch = self.torch
        encoded = [prompt_text(row, self.tokenizer, limit=self.context_limit)[1] for row in rows]
        if not encoded:
            raise ValueError("空批次")
        size = max(map(len, encoded))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("需要 pad 或 eos token")
        ids = torch.full((len(rows), size), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(encoded):
            ids[i, -len(seq):] = torch.tensor(seq, device=self.device)
            mask[i, -len(seq):] = 1
        return {"input_ids": ids, "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp_min(0)}, sum(map(len, encoded))

    def logits(self, rows):
        inputs, count = self._inputs(rows)
        return self.model(**inputs, use_cache=False, num_logits_to_keep=1).logits[:, -1, :].float(), count

    def select(self, row, selection="greedy", temperature=1., generator=None, seed=None):
        torch = self.torch
        if selection not in ("sample", "greedy") or not temperature > 0:
            raise ValueError("选择方式或温度无效")
        with torch.no_grad():
            scores, _ = self.logits([row])
            legal = scores[0, self.ids[:action_count(row)]] / temperature
            logp = legal.log_softmax(-1)
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            index = int(legal.argmax()) if selection == "greedy" else int(torch.multinomial(logp.exp(), 1, generator=generator))
        return {"index": index, "token": TOKENS[index],
                "logprob": float(logp[index]) if selection == "sample" else 0.,
                "policy_logprob": float(logp[index]), "selection": selection, "temperature": temperature}

    def __call__(self, row):
        return self.select(row)["index"]

    def generate_action(self, row):
        from transformers import LogitsProcessor, LogitsProcessorList
        torch, allowed = self.torch, self.ids[:action_count(row)]

        class LegalOnly(LogitsProcessor):
            def __call__(self, input_ids, scores):
                result = torch.full_like(scores, -float("inf"))
                result[:, allowed] = scores[:, allowed]
                return result

        inputs, _ = self._inputs([row])
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                logits_processor=LogitsProcessorList([LegalOnly()]),
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                eos_token_id=None)
        token_id = int(output[0, -1])
        if token_id not in allowed:
            raise ValueError("生成了非法动作")
        return self.ids.index(token_id)

    def save(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(path, save_embedding_layers=False)
        self.tokenizer.save_pretrained(path)
        rows = {"input_rows": self.model.get_input_embeddings().weight[self.ids].detach().cpu().contiguous()}
        if not self.tied:
            rows["output_rows"] = self.model.get_output_embeddings().weight[self.ids].detach().cpu().contiguous()
        save_file(rows, str(path / "action_rows.safetensors"))
        (path / "student.json").write_text(json.dumps({"format": FORMAT, "base": self.base,
            "revision": self.revision, "action_token_ids": self.ids, "action_tokens": TOKENS,
            "original_vocab": self.original_vocab, "base_tokenizer_size": self.base_tokenizer_size,
            "context_limit": self.context_limit,
            "tied": self.tied, "initialization": "mean_noise_0.001/v1",
            "rank": 16, "alpha": 32, "dropout": .05}, indent=2), encoding="utf-8")

    def export_merged(self, path):
        # 在副本上合并，避免破坏正在训练的 LoRA。
        import tempfile
        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "adapter"
            self.save(checkpoint)
            restored = Student(adapter=checkpoint, device=self.device)
            merged = restored.model.merge_and_unload()
            output = Path(path)
            output.mkdir(parents=True, exist_ok=False)
            merged.save_pretrained(output)
            self.tokenizer.save_pretrained(output)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="将动作 LoRA 适配器导出为标准合并因果语言模型")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    Student(adapter=args.adapter, device=args.device).export_merged(args.out)


if __name__ == "__main__":
    main()
