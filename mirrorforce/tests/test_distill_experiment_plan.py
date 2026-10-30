"""云端编排使用固定WinBot及未消费的正式测试发牌。"""
from pathlib import Path
from mirrorforce.distill.experiment import evaluation_command


def test_windbot_evaluation_has_no_teacher_process():
    config = dict(windbot_exe='WindBot.exe', deck='deck', cards_db='cards', script_root='assets',
                  native='native', prompt_style='named_menu')
    command = evaluation_command(config, ['python', '-m'], Path('/runs'), 'sft', '/checkpoint')
    assert 'mirrorforce.distill.arena' in command and '--teacher' not in command
    assert command[command.index('--seed') + 1] == '300100'
    assert command[command.index('--pairs') + 1] == '50'
    assert command[command.index('--prompt-style') + 1] == 'named_menu'
    assert command[command.index('--rules') + 1] == str(Path('/runs/demonstrations/manifest.json'))


def test_teacher_reference_requires_explicit_alternative_path():
    assert evaluation_command({}, [], Path('/runs'), 'sft', '/checkpoint') is None
