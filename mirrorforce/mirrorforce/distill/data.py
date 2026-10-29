"""v2 客户端可见信息契约、语义化输入和确定性公开历史。"""
import copy
import hashlib
import json
import math
import re
import sqlite3
import struct
from pathlib import Path
from mirrorforce.netduel import constants as C

SCHEMA = "mirrorforce_action_distill/v2"
SERIALIZER_VERSION = "semantic_table_json/v3"
TOKENS = [f"<|action_{i:03d}|>" for i in range(192)]
ACTION_FIELDS = ("act", "phase", "finish", "code", "effect", "position", "number", "place", "attribute", "spec")
EXACT_ACTION_FIELDS = ("response", "option_desc", "option_code", "option_index", "exact_semantics_version", "description_code")
LOCATIONS = {1: "deck", 2: "hand", 4: "monster", 8: "spell_trap", 16: "graveyard", 32: "banished", 64: "extra"}
PHASES = {0: "unknown", 1: "draw", 2: "standby", 4: "main_phase_1", 8: "battle_start", 16: "battle_step", 32: "damage",
          64: "damage_calculation", 128: "battle", 256: "main_phase_2", 512: "end"}
POSITIONS = {0: "unknown", 1: "face_up_attack", 2: "face_down_attack", 3: "attack", 4: "face_up_defense", 5: "face_up", 8: "face_down_defense", 10: "face_down"}
ACTS = {0: "select", 1: "set_spell_trap", 2: "change_position", 3: "special_summon", 4: "normal_summon", 5: "set_monster",
        6: "attack", 7: "direct_attack", 8: "activate", 9: "cancel"}
ACTION_PHASES = {0: None, 1: "battle", 2: "main_phase_2", 3: "end"}
PROMPTS = {v: k[4:].lower() for k, v in vars(C).items() if k.startswith("MSG_SELECT_") or k.startswith("MSG_ANNOUNCE_")}
INPUT_FIELDS = ("observation", "card_dictionary", "public_history", "legal_actions")
REQUIRED_METADATA = ("run_id", "game_id", "decision_id", "subdecision_id", "deal_seed", "first_player", "viewer", "deck_id",
                     "opponent_deck_id", "teacher_identity", "obs_sha256", "student_checkpoint_sha", "source_commit", "engine_hash", "cards_db_hash", "tokenizer_hash")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def probabilities(logits):
    if not 1 <= len(logits) <= 192 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in logits):
        raise ValueError("动作数超限或教师分数非有限值")
    values = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(values) for x in values]


def role(player, viewer):
    if player not in (0, 1) or viewer not in (0, 1):
        raise ValueError("座位必须为 0 或 1")
    return "self" if player == viewer else "opponent"


def location_ref(player, location, sequence, viewer):
    base = location & 127
    if base not in LOCATIONS or sequence < 0:
        raise ValueError("未知区域或负数位置")
    return f"{role(player, viewer)}.{LOCATIONS[base]}.{sequence}"


def public_view(client, rows):
    """只接收客户端观察器；未知卡仅保留位置，禁止读取其身份与数值。"""
    board, cards = client.board, []
    for (player, location), zone in sorted(board.zones.items()):
        if location == C.LOCATION_DECK:
            continue
        for sequence, card in enumerate(zone):
            if card is None:
                continue
            item = {"player": int(player), "location": int(location), "sequence": sequence}
            if card.known:
                item.update({key: int(getattr(card, key)) for key in ("code", "position", "attack", "defense", "level", "rank", "status")})
                item.update({key: int(getattr(card, key, 0)) for key in
                             ("type", "attribute", "race", "lscale", "rscale", "link", "link_marker")})
                key = (player, location, sequence)
                item["materials"] = [int(c.code) if c.known else 0 for c in board.materials.get(key, [])]
                item["counters"] = dict(board.counters.get(key, {}))
            else:
                item["code"] = 0
            cards.append(item)
    actions = [{**{key: row[key] for key in ACTION_FIELDS},
                **{key: row[key] for key in EXACT_ACTION_FIELDS if key in row}} for row in rows]
    probabilities([0.] * len(actions))
    return {"viewer": int(client.seat), "turn_player": int(board.turn_player), "phase": int(board.phase),
            "deck_counts": list(board.deck_count), "extra_counts": list(board.extra_count),
            "chain": list(getattr(getattr(board, "disclosure", None), "_chain_stack", [])),
            "equip_targets": [[list(k), list(v)] for k, v in sorted(getattr(board, "equip_targets", {}).items())],
            "card_targets": [[list(k), [list(v) for v in sorted(values)]] for k, values in sorted(getattr(board, "card_targets", {}).items())],
            "cards": cards, "actions": actions}


class CardTexts:
    def __init__(self, path):
        with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(texts)")}
            options = [f"str{i}" for i in range(1, 17) if f"str{i}" in columns]
            selected = ",".join(["id", "name", "desc"] + options)
            self.texts = {}
            for code, name, text, *values in db.execute(f"SELECT {selected} FROM texts"):
                self.texts[int(code)] = {"name": name or "", "effect_text": text or "",
                    "effect_options": {str(int(column[3:]) - 1): value for column, value in zip(options, values) if value}}

    def enrich(self, view, history=None):
        codes = {int(r["code"]) for r in view["cards"] + view["actions"] if r["code"]}
        codes.update(int(x) for c in view["cards"] for x in c.get("materials", []) if x)
        codes.update(int(x) for x in view.get("chain", []) if x)
        codes.update(int(a["description_code"]) >> 4 for a in view["actions"] if a.get("description_code") is not None and int(a["description_code"]) >= 10000)
        codes.update(history_codes(history or []))
        if codes - self.texts.keys():
            raise ValueError(f"可见卡牌缺少数据库文本：{sorted(codes - self.texts.keys())}")
        return {**copy.deepcopy(view), "card_texts": {str(code): copy.deepcopy(self.texts[code]) for code in sorted(codes)}}


def semantic_action(action, index, prompt_msg=0):
    if not 0 <= index < 192:
        raise ValueError("动作索引超限")
    act, phase = int(action["act"]), int(action["phase"])
    if act not in ACTS or phase not in ACTION_PHASES:
        raise ValueError("未知动作类型或阶段编码")
    kind = "finish_selection" if action["finish"] else ACTS[act]
    if phase:
        kind = "change_phase"
    if kind == "select":
        kind = PROMPTS.get(prompt_msg, "unknown_selection")
    result = {"token": TOKENS[index], "kind": kind, "description": kind.replace("_", " ")}
    spec = str(action["spec"])
    if spec:
        # 原生 ls_to_spec 对牌堆区域没有字母前缀；数字是本次获准选择的牌堆候选引用。
        if spec.isdigit() and int(spec) > 0:
            result["candidate_ref"] = f"self.offered_deck_card.{int(spec) - 1}"
            spec = ""
    if spec:
        match = re.fullmatch(r"(o?)([hmsgrx])(\d+)([a-z]?)", spec)
        if not match or int(match[3]) < 1:
            raise ValueError(f"未知原生卡牌引用：{spec}")
        zone = {"h": "hand", "m": "monster", "s": "spell_trap", "g": "graveyard", "r": "banished", "x": "extra"}[match[2]]
        ref_field = "source_ref" if act else "candidate_ref"
        result[ref_field] = f"{'opponent' if match[1] else 'self'}.{zone}.{int(match[3]) - 1}"
        if match[4]:
            result[ref_field] += f".material.{ord(match[4]) - ord('a')}"
    if action["code"]:
        result["card_code"] = int(action["code"])
    if phase:
        result["target_phase"] = ACTION_PHASES[phase]
    # 此索引不等于效果正文段落编号，不能凭数值编造解释。
    result["native_description_index"] = int(action["effect"])
    if "description_code" in action and action["description_code"] is not None:
        result["description_code"] = int(action["description_code"])
    elif act == 8:
        # 旧原生版本缺少精确描述，明确标注不可用，不能以 effect_ 伪装原始 desc。
        result["description_status"] = "unavailable_in_native_build"
    if prompt_msg == C.MSG_SELECT_OPTION and "description_code" not in result:
        raise ValueError("效果选项提示需要包含精确 description_code 的新版原生模块")
    if prompt_msg in (C.MSG_ANNOUNCE_NUMBER, C.MSG_ANNOUNCE_RACE):
        if action.get("exact_semantics_version") != 1 or "response" not in action:
            raise ValueError("该提示需要包含精确 response 的新版原生模块")
        if prompt_msg == C.MSG_ANNOUNCE_NUMBER:
            result["announced_number"] = int(action["response"])
        else:
            result["announced_race_mask"] = int(action["response"])
    if action["position"]:
        result["position"] = POSITIONS.get(int(action["position"]), f"unknown_position_{action['position']}")
    for key in ("number", "place", "attribute"):
        if action[key]:
            result[f"native_{key}"] = int(action[key])
    place = int(action["place"])
    if place:
        if 1 <= place <= 7:
            result["destination_ref"] = f"self.monster.{place - 1}"
        elif 8 <= place <= 15:
            result["destination_ref"] = f"self.spell_trap.{place - 8}"
        elif 16 <= place <= 22:
            result["destination_ref"] = f"opponent.monster.{place - 16}"
        elif 23 <= place <= 30:
            result["destination_ref"] = f"opponent.spell_trap.{place - 23}"
        else:
            result["destination_ref"] = f"unknown_place_{place}"
    if action["attribute"]:
        flags = int(action["attribute"])
        result["attributes"] = [name for bit, name in ((1,"earth"),(2,"water"),(4,"fire"),(8,"wind"),(16,"light"),(32,"dark"),(64,"divine")) if flags & bit]
    return result


def semantic_view(view, history=None):
    """从原始客户端视图投影语义白名单，不复制训练监督字段。"""
    viewer, phase = int(view["viewer"]), int(view["phase"])
    if phase not in PHASES:
        raise ValueError("未知当前阶段")
    cards = []
    for card in view["cards"]:
        if card["location"] == C.LOCATION_DECK:
            raise ValueError("盘面不能包含牌序")
        item = {"ref": location_ref(card["player"], card["location"], card["sequence"], viewer),
                "visibility": "known" if card["code"] else "unknown"}
        if card["code"]:
            item.update(card_code=int(card["code"]), position=POSITIONS.get(int(card.get("position", 0)), "unknown"))
            for field in ("attack", "defense", "level", "rank", "status", "type", "attribute", "race", "lscale", "rscale", "link", "link_marker"):
                if field in card:
                    item[field] = int(card[field])
            item["materials"] = [int(x) for x in card.get("materials", [])]
            item["counters"] = {str(k): int(v) for k, v in sorted(card.get("counters", {}).items(), key=lambda p: str(p[0]))}
        cards.append(item)
    observation = {"turn": int(view.get("turn", 0)), "turn_player": role(view["turn_player"], viewer), "phase": PHASES[phase],
                   "cards": cards, "prompt_type": PROMPTS.get(int(view.get("prompt_msg", 0)), "unknown_selection"),
                   "deck_counts": {role(p, viewer): int(view["deck_counts"][p]) for p in (0, 1)},
                   "extra_counts": {role(p, viewer): int(view["extra_counts"][p]) for p in (0, 1)},
                   "chain": [{"link": i + 1, "card_code": int(code)} for i, code in enumerate(view.get("chain", []))]}
    if "lp" in view:
        observation["lp"] = {role(p, viewer): int(view["lp"][p]) for p in (0, 1)}
    observation["equip_targets"] = [[location_ref(*s, viewer), location_ref(*t, viewer)] for s, t in view.get("equip_targets", [])]
    observation["card_targets"] = [[location_ref(*s, viewer), [location_ref(*t, viewer) for t in ts]] for s, ts in view.get("card_targets", [])]
    codes = {c["card_code"] for c in cards if "card_code" in c}
    codes.update(x for c in cards for x in c.get("materials", []) if x)
    codes.update(int(a["code"]) for a in view["actions"] if a["code"])
    codes.update(int(a["description_code"]) >> 4 for a in view["actions"] if a.get("description_code") is not None and int(a["description_code"]) >= 10000)
    codes.update(x["card_code"] for x in observation["chain"] if x["card_code"])
    codes.update(history_codes(history or []))
    texts = view.get("card_texts", {})
    dictionary = {str(code): {"name": texts[str(code)].get("name", ""), "effect_text": texts[str(code)].get("effect_text", texts[str(code)].get("effect", "")),
                              "effect_options": texts[str(code)].get("effect_options", {})}
                  for code in sorted(codes) if str(code) in texts}
    actions = [semantic_action(a, i, int(view.get("prompt_msg", 0))) for i, a in enumerate(view["actions"])]
    for action in actions:
        desc = action.get("description_code")
        if desc is None:
            continue
        if desc >= 10000:
            effect = texts.get(str(desc >> 4), {}).get("effect_options", {}).get(str(desc & 15))
            if effect:
                action["description"] = effect
                action["description_status"] = "exact_card_description"
            else:
                action["description_status"] = "exact_code_text_unavailable"
        else:
            action["description_status"] = "system_description_text_unavailable"
    return {"observation": observation, "card_dictionary": dictionary, "public_history": copy.deepcopy(history or []),
            "legal_actions": actions}


def model_input(row):
    if all(k in row for k in INPUT_FIELDS):
        return {k: copy.deepcopy(row[k]) for k in INPUT_FIELDS}
    return semantic_view(row["view"], row.get("public_history", row.get("history", [])))


def compact_input(inputs):
    """可逆表格与共享文本字典；不删除任何语义字段或效果正文。"""
    from collections import Counter
    counts = Counter()
    reserved = {"$text", "$table", "$missing"}

    def scan(value):
        if isinstance(value, str) and len(value) >= 40:
            counts[value] += 1
        elif isinstance(value, list):
            for item in value:
                scan(item)
        elif isinstance(value, dict):
            if reserved & value.keys():
                raise ValueError("输入使用了紧凑编码保留键")
            for item in value.values():
                scan(item)
    scan(inputs)
    pool = sorted(text for text, count in counts.items() if count > 1)
    indexes = {text: i for i, text in enumerate(pool)}
    missing = {"$missing": True}

    def encode(value):
        if isinstance(value, str) and value in indexes:
            return {"$text": indexes[value]}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in sorted(value.items())}
        if isinstance(value, list):
            if len(value) >= 2 and all(isinstance(item, dict) for item in value):
                columns = sorted({key for row in value for key in row})
                if columns:
                    raw = [[encode(row[key]) if key in row else missing for key in columns] for row in value]
                    defaults = []
                    for col in range(len(columns)):
                        frequencies = Counter(json.dumps(row[col], sort_keys=True, ensure_ascii=False) for row in raw)
                        selected = sorted(frequencies, key=lambda key: (-frequencies[key], key))[0]
                        defaults.append(json.loads(selected))
                    same = lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
                    rows = [[part for index, cell in enumerate(row) if not same(cell, defaults[index]) for part in (index, cell)] for row in raw]
                    table = {"$table": {"columns": columns, "defaults": defaults, "rows": rows}}
                    dense = {"$table": {"columns": columns, "dense_rows": raw}}
                    plain = [{key: encode(item) for key, item in sorted(row.items())} for row in value]
                    # 小表格保留可读对象，避免表头比原文更大。
                    measure = lambda obj: len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
                    return min((table, dense, plain), key=measure)
            return [encode(item) for item in value]
        return value
    return {"format": SERIALIZER_VERSION, "text_pool": pool, "input": encode(inputs)}


def decode_compact_input(payload):
    """验收用逆变换；结果必须精确等于原语义对象。"""
    if payload.get("format") != SERIALIZER_VERSION:
        raise ValueError("紧凑输入版本不匹配")
    pool = payload["text_pool"]
    missing = {"$missing": True}

    def decode(value):
        if isinstance(value, list):
            return [decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"$text"}:
            index = value["$text"]
            if type(index) is not int or not 0 <= index < len(pool):
                raise ValueError("共享文本索引非法")
            return pool[index]
        if set(value) == {"$table"}:
            table = value["$table"]
            columns = table["columns"]
            if "dense_rows" in table:
                if len(set(columns)) != len(columns) or any(len(row) != len(columns) for row in table["dense_rows"]):
                    raise ValueError("稠密表格列定义非法")
                return [{column: decode(cell) for column, cell in zip(columns, row) if cell != missing} for row in table["dense_rows"]]
            defaults = table["defaults"]
            if len(columns) != len(defaults) or len(set(columns)) != len(columns):
                raise ValueError("表格列定义非法")
            rows = []
            for sparse in table["rows"]:
                if len(sparse) % 2:
                    raise ValueError("表格覆盖值必须为索引和值对")
                cells, seen = copy.deepcopy(defaults), set()
                for i in range(0, len(sparse), 2):
                    index = sparse[i]
                    if type(index) is not int or not 0 <= index < len(columns) or index in seen:
                        raise ValueError("表格覆盖列索引非法")
                    seen.add(index)
                    cells[index] = sparse[i + 1]
                rows.append({column: decode(cell) for column, cell in zip(columns, cells) if cell != missing})
            return rows
        return {key: decode(item) for key, item in value.items()}
    return decode(payload["input"])


def prompt_details(row, tokenizer, limit=4096):
    """只裁剪最旧公开历史，绝不裁剪盘面和菜单，不修改原始数据。"""
    inputs = model_input(row)
    original_history = len(inputs["public_history"])
    history = inputs["public_history"]

    def encode(removed):
        inputs["public_history"] = history[removed:]
        content = json.dumps(compact_input(inputs), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": "根据本方可见盘面、公开历史和合法菜单选择动作。只输出一个合法动作标记，不输出解释。输入采用可逆紧凑JSON：$text索引引用text_pool完整正文。$table的columns按0起始编号；dense_rows按列顺序存值；或defaults存各列默认值，rows每行是[列号,覆盖值,...]，未覆盖列用默认值。$missing表示字段不存在。self/opponent表示本方/对手，位置ref末尾数字从0计数；offered_deck_card仅为本次获准选择的牌堆候选，不代表公开牌序。"},
            {"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        return text, ids

    text, ids = encode(0)
    if len(ids) + 1 <= limit:
        return text, ids, {"history_removed": 0, "input_tokens": len(ids)}
    # 先证明必需字段能够装下，避免为必然超长盘面逐条重复tokenize整个历史。
    best_text, best_ids = encode(original_history)
    if len(best_ids) + 1 > limit:
        raise ValueError("context_overflow：盘面和完整菜单超过上下文上限")
    low, high, removed = 1, original_history, original_history
    while low <= high:
        middle = (low + high) // 2
        candidate_text, candidate_ids = encode(middle)
        if len(candidate_ids) + 1 <= limit:
            removed, best_text, best_ids = middle, candidate_text, candidate_ids
            high = middle - 1
        else:
            low = middle + 1
    return best_text, best_ids, {"history_removed": removed, "input_tokens": len(best_ids)}


def prompt_text(row, tokenizer, limit=4096):
    text, ids, _ = prompt_details(row, tokenizer, limit)
    return text, ids


def history_codes(history):
    """仅解析语义事件的显式卡牌字段，不从任意数值猜测卡号。"""
    out = set()
    for event in history:
        if event.get("card_code"):
            out.add(int(event["card_code"]))
        for card in event.get("cards", []):
            if card.get("card_code"):
                out.add(int(card["card_code"]))
        action = event.get("action", {})
        if action.get("card_code"):
            out.add(int(action["card_code"]))
    return out


def build_sample(view, history, metadata, teacher_logits, chosen, behavior=None):
    """构造未验收样本；整局成功后由采集器标记 completed/training_eligible。"""
    missing = set(REQUIRED_METADATA) - metadata.keys()
    if missing:
        raise ValueError(f"缺少样本元数据：{sorted(missing)}")
    behavior = behavior or {}
    n = len(view["actions"])
    probabilities(teacher_logits)
    if len(teacher_logits) != n or type(chosen) is not int or not 0 <= chosen < n:
        raise ValueError("菜单、分数或动作未对齐")
    return {"schema": SCHEMA, "serializer_version": SERIALIZER_VERSION, "boundary": "client_public_observation",
            **copy.deepcopy(metadata), "split": split_for(metadata["deal_seed"]), **semantic_view(view, history),
            "view": copy.deepcopy(view), "history": copy.deepcopy(history),
            "action_mapping": [{"token": TOKENS[i], "native_index": i} for i in range(n)], "menu_sha256": digest(view["actions"]),
            "teacher_logits": list(teacher_logits), "chosen": chosen, "chosen_index": chosen, "chosen_token": TOKENS[chosen],
            "behavior_logprob": behavior.get("behavior_logprob"), "selection": behavior.get("selection", "greedy"),
            "temperature": behavior.get("temperature", 1.),
            **{k: behavior.get(k) for k in ("teacher_ms", "student_ms", "serialization_ms", "input_tokens")},
            "completed": False, "training_eligible": False, "episode_result": None}


def validate(row):
    if row.get("schema") != SCHEMA or row.get("serializer_version") != SERIALIZER_VERSION or row.get("boundary") != "client_public_observation":
        raise ValueError("样本来源或版本不匹配；v1 必须重新采集或显式迁移")
    missing = set(REQUIRED_METADATA + INPUT_FIELDS) - row.keys()
    if missing:
        raise ValueError(f"v2 样本缺少字段：{sorted(missing)}")
    if row.get("split") != split_for(row["deal_seed"]):
        raise ValueError("种子与锁定划分不匹配")
    if row["viewer"] not in (0, 1) or row["first_player"] not in ("self", "opponent"):
        raise ValueError("观察者或先手角色非法")
    if not all(isinstance(row[k], (str, int)) and not isinstance(row[k], bool) and str(row[k]) for k in REQUIRED_METADATA):
        raise ValueError("元数据必须为非空字符串或整数")
    if row.get("completed") is not True or row.get("training_eligible") is not True or not row.get("episode_result"):
        raise ValueError("样本不是通过整局验收的可训练数据")
    n = len(row["legal_actions"])
    if n < 2 or len(row["teacher_logits"]) != n:
        raise ValueError("样本必须包含对齐的非强制动作")
    probabilities(row["teacher_logits"])
    chosen = row.get("chosen_index")
    if type(chosen) is not int or not 0 <= chosen < n or row.get("chosen") != chosen or row.get("chosen_token") != TOKENS[chosen]:
        raise ValueError("动作索引或标记非法")
    if row.get("action_mapping") != [{"token": TOKENS[i], "native_index": i} for i in range(n)]:
        raise ValueError("动作映射错位")
    if any(a.get("token") != TOKENS[i] for i, a in enumerate(row["legal_actions"])):
        raise ValueError("菜单 token 与映射不同步")
    if row.get("view") is not None:
        if row.get("menu_sha256") != digest(row["view"]["actions"]):
            raise ValueError("菜单摘要不匹配")
        if model_input(row) != semantic_view(row["view"], row.get("public_history", [])):
            raise ValueError("语义输入与审计副本不一致")
    if row.get("selection") not in ("sample", "greedy"):
        raise ValueError("未知行为策略")
    if row["selection"] == "sample" and (not isinstance(row.get("behavior_logprob"), (float, int)) or not math.isfinite(row["behavior_logprob"]) or row["behavior_logprob"] > 0):
        raise ValueError("采样必须记录有限非正行为 log-prob")
    return row


def read_rows(path, split="train"):
    if split not in ("train", "validation", "test"):
        raise ValueError("未知划分")
    rows = [validate(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    pairs, games, decisions, traces = {}, {}, set(), {}
    for row in rows:
        pair, game = (row["deck_id"], row["opponent_deck_id"], row["deal_seed"]), row["game_id"]
        if (pair in pairs and pairs[pair] != row["split"]) or (game in games and games[game] != row["split"]):
            raise ValueError("相同发牌或整局进入不同划分")
        pairs[pair], games[game] = row["split"], row["split"]
        decision = (game, row["viewer"], row["decision_id"], row["subdecision_id"])
        if decision in decisions:
            raise ValueError("重复决策轨迹")
        decisions.add(decision)
        trace = digest(model_input(row))
        if trace in traces and traces[trace] != row["split"]:
            raise ValueError("重复状态轨迹跨划分")
        traces[trace] = row["split"]
    return [row for row in rows if row["split"] == split]


def split_for(seed):
    if type(seed) is not int:
        raise ValueError("种子必须为整数")
    if 300000 <= seed < 301000:
        return "test"
    if 200000 <= seed < 201000:
        return "validation"
    if 100000 <= seed < 110000:
        return "train"
    raise ValueError("种子不在锁定的训练/验证/测试区间")


def public_event(msg, body, viewer):
    """解析已路由给当前客户端的消息白名单；绝不保存未经解释的包字节。"""
    msg, body = int(msg), bytes(body)
    def need(n):
        if len(body) < n:
            raise ValueError(f"公开事件 {msg} 长度不足")
    if msg == C.MSG_NEW_TURN:
        need(1)
        return {"event": "new_turn", "player": role(body[0], viewer)}
    if msg == C.MSG_NEW_PHASE:
        need(2)
        value = struct.unpack_from("<H", body)[0]
        return {"event": "new_phase", "phase": PHASES.get(value, f"unknown_phase_{value}")}
    if msg in (C.MSG_DAMAGE, C.MSG_RECOVER, C.MSG_PAY_LPCOST, C.MSG_LPUPDATE):
        need(5)
        return {"event": {91: "damage", 92: "recover", 100: "pay_lp", 94: "lp_update"}[msg], "player": role(body[0], viewer), "amount": struct.unpack_from("<I", body, 1)[0]}
    if msg == C.MSG_DRAW:
        need(2)
        return {"event": "draw", "player": role(body[0], viewer), "count": body[1]}
    if msg in (C.MSG_CONFIRM_CARDS, C.MSG_CONFIRM_DECKTOP, C.MSG_CONFIRM_EXTRATOP):
        head = 3 if msg == C.MSG_CONFIRM_CARDS else 2
        need(head)
        count = body[head - 1]
        need(head + count * 7)
        cards = []
        for i in range(count):
            at = head + i * 7
            code = struct.unpack_from("<I", body, at)[0] & 0x7fffffff
            cards.append({"card_code": code, "ref": location_ref(body[at + 4], body[at + 5], body[at + 6], viewer)})
        return {"event": "reveal_cards", "cards": cards}
    if msg in (C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING):
        need(8)
        return {"event": {60: "normal_summon", 62: "special_summon", 64: "flip_summon"}[msg], "card_code": struct.unpack_from("<I", body)[0] & 0x7fffffff,
                "ref": location_ref(body[4], body[5], body[6], viewer)}
    if msg == C.MSG_MOVE:
        need(16)
        code, old, new, reason = struct.unpack_from("<IIII", body)
        def ref(packed):
            return location_ref(packed & 255, (packed >> 8) & 255, (packed >> 16) & 255, viewer) if (packed >> 8) & 255 else "outside_duel"
        return {"event": "move", "card_code": code & 0x7fffffff, "from": ref(old), "to": ref(new), "reason_code": reason}
    if msg == C.MSG_ATTACK:
        need(8)
        source, target = struct.unpack_from("<II", body)
        return {"event": "attack", "source": location_ref(source & 255, (source >> 8) & 255, (source >> 16) & 255, viewer),
                "target": "direct" if not target else location_ref(target & 255, (target >> 8) & 255, (target >> 16) & 255, viewer)}
    if msg == C.MSG_CHAINING:
        need(16)
        return {"event": "activate", "card_code": struct.unpack_from("<I", body)[0] & 0x7fffffff,
                "source": location_ref(body[4], body[5], body[6], viewer), "description_code": struct.unpack_from("<I", body, 11)[0], "link": body[15]}
    if msg in (C.MSG_CHAINED, C.MSG_CHAIN_SOLVING, C.MSG_CHAIN_SOLVED, C.MSG_CHAIN_NEGATED, C.MSG_CHAIN_DISABLED):
        need(1)
        return {"event": {71: "chain_added", 72: "chain_solving", 73: "chain_solved", 75: "chain_negated", 76: "chain_disabled"}[msg], "link": body[0]}
    if msg == C.MSG_CHAIN_END:
        return {"event": "chain_end"}
    if msg == C.MSG_HINT:
        need(6)
        if body[0] in (C.HINT_SELECTMSG, C.HINT_EFFECT, C.HINT_MESSAGE):
            return {"event": "public_hint", "hint_type": {3: "selection", 6: "effect", 2: "message"}[body[0]], "description_code": struct.unpack_from("<I", body, 2)[0]}
    return None
