"""固定WinBot对手的独立学生开发评测；无需加载教师网络。"""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for field in ['rules', 'adapter', 'deck', 'cards-db', 'script-root', 'native', 'windbot-exe', 'out']:
        p.add_argument('--' + field, required=True, type=Path)
    p.add_argument('--seed', type=int, default=200100)
    p.add_argument('--pairs', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--prompt-style', choices=['compact', 'named_menu'])
    p.add_argument('--game-seconds', type=float, default=900)
    a = p.parse_args(argv)
    from .data import split_for, CardTexts, SERIALIZER_VERSION
    from .stages import source_identity, file_hash
    from .student import Student
    from .policy import StudentPolicy
    from .windbot import WindBotPolicy
    from .collect import content_identity
    from .evaluate import summarize
    from tools.mf_runtime_policy_service import load_native
    from tools.mf_runtime_client_parity import host_core
    from tools.mf_runtime_local_match import play_game
    from mirrorforce.netduel.cards import CardPool
    from mirrorforce.worldmodel.engine import load_ydk
    if a.pairs <= 0 or a.game_seconds <= 0 or split_for(a.seed) not in ('validation', 'test') or split_for(a.seed) != split_for(a.seed+a.pairs-1):
        raise ValueError('评测必须使用独立验证或测试种子区间')
    rules = json.loads(a.rules.read_text())
    identity = rules.get('teacher_identity', rules)
    if identity['serving_native_sha256'] != file_hash(a.native) or identity['cards_db_sha256'] != file_hash(a.cards_db):
        raise ValueError('规则资源与客户端观察器身份不匹配')
    a.out.mkdir(parents=True, exist_ok=False)
    native, _ = load_native(SimpleNamespace(script_root=a.script_root,cards_db=a.cards_db,
        code_list=a.cards_db.parent/'code_list.txt',backend='uniform',
        announce_tables=next(a.cards_db.parent.glob('announce-tables-*.json')),announce_cap=192))
    model = Student(adapter=a.adapter, device=a.device, context_limit=16384, prompt_style=a.prompt_style)
    checkpoint_sha = content_identity(a.adapter)
    deck_sha = file_hash(a.deck)
    manifest = {**vars(a), **source_identity(), 'deck_sha256':deck_sha,
        'cards_db_sha256':file_hash(a.cards_db),'native_sha256':file_hash(a.native),
        'teacher_identity':identity,'pairing':'role_deck_orders_preserved/v1',
        'serializer_version':SERIALIZER_VERSION,'selection':'greedy','context_limit':model.context_limit,
        'prompt_style':model.prompt_style,
        'prompt_version':model.prompt_version,
        'student_checkpoint_sha':checkpoint_sha,
        'opponent_identity':{'kind':'windbot','executor':'SkyStriker','seed_control':'WINDBOT_SEED',
            'files':{str(f.relative_to(a.windbot_exe.parent)):file_hash(f)
                     for f in sorted(a.windbot_exe.parent.rglob('*')) if f.is_file() and f.suffix in ('.exe','.dll','.json','.config')}}}
    (a.out/'manifest.json').write_text(json.dumps(manifest,default=str,indent=2))
    cards, texts, recipe = CardPool(a.cards_db), CardTexts(a.cards_db), load_ydk(a.deck)
    rows = []
    with tempfile.TemporaryDirectory() as scratch, (a.out/'games.jsonl').open('w') as output:
        core = host_core(a.native,a.cards_db,a.script_root,scratch)
        for seed in range(a.seed,a.seed+a.pairs):
            for first in (True,False):
                students = []
                def factory(role,seat):
                    if role == 'b':
                        return WindBotPolicy(identity,a.windbot_exe,a.deck,a.cards_db,seed=seed)
                    metadata = {'run_id':a.out.name,'game_id':f'{seed}-{int(first)}','deal_seed':seed,
                        'first_player':'self' if first else 'opponent','deck_id':deck_sha,
                        'opponent_deck_id':deck_sha,'engine_hash':manifest['native_sha256'],
                        'cards_db_hash':manifest['cards_db_sha256'],'source_commit':manifest['source_commit'],
                        'student_checkpoint_sha':checkpoint_sha,'tokenizer_hash':__import__('mirrorforce.distill.data',fromlist=['digest']).digest(model.tokenizer.get_vocab())}
                    student = StudentPolicy(native,identity,chooser=model,texts=texts,metadata=metadata,seed=seed)
                    students.append(student)
                    return student
                row=play_game(core,cards,recipe,seed,first,{}, {},max_seconds=a.game_seconds,
                    policy_factory=factory,preserve_role_deals=True)
                row.update(teacher_queries=0,
                    decision_trace=[r for s in students for r in s.decisions],
                    own_actions=[r for s in students for r in s.history if r.get('event') == 'own_decision'],
                    latency_ms=[r['student_ms'] for s in students for r in s.decisions if r.get('model_called')])
                rows.append(row)
                output.write(json.dumps(row)+'\n')
                output.flush()
                print(json.dumps({k:v for k,v in row.items() if k not in ('latency_ms','decision_trace','own_actions')}),flush=True)
    result=summarize(rows,a.pairs*2)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    return int(result['failures'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
