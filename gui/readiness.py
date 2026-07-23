"""起動前 readiness の評価 (ADR-0045)。

「何が必須か」の権威は vspeech.preflight (ADR-0038) にあり、この module は
それを呼んで表示用に整形するだけ。必須項目の知識をここへ複製しないこと —
複製すると GUI が緑なのに起動して落ちる (逆も) という形で drift が出る。
"""

from dataclasses import dataclass

from vspeech.config import Config
from vspeech.exceptions import ConfigProblem
from vspeech.preflight import collect_problems

# config に <name>.enable を持つ worker。**この列挙は preflight._CHECKERS と
# 1:1 で追随させること** — 片方だけ増やすと readiness が黙って過少報告する
# (stream_vc を落としていたときは「有効な worker がありません」と出た)。
# `.enable` を持つ config セクションを機械的に集める案は採らない: telemetry の
# ような worker でないセクションも `.enable` を持つため。
WORKER_NAMES: tuple[str, ...] = (
    "recording",
    "transcription",
    "translation",
    "tts",
    "vc",
    "playback",
    "subtitle",
    "stream_vc",
)

# stream_vc は EventType の鎖に乗らない独立サブシステム (ADR-0050): マイク →
# 固定ブロック VC → 出力 が閉じた 1 本で、routes_list にも text_send_operations
# にも現れない。表示用にその 1 本を固定文言で持つ。
STREAM_VC_FLOW: list[str] = ["(mic)", "stream_vc", "(出力)"]


@dataclass(frozen=True)
class WorkerReadiness:
    worker: str
    problems: list[ConfigProblem]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class Readiness:
    workers: list[WorkerReadiness]
    flow: list[list[str]]
    error: str | None = None
    """readiness の評価自体が失敗した理由 (成立していれば None)。"""

    @property
    def ok(self) -> bool:
        return self.error is None and all(worker.ok for worker in self.workers)

    @property
    def problem_count(self) -> int:
        return sum(len(worker.problems) for worker in self.workers)


def enabled_workers(config: Config) -> list[str]:
    return [name for name in WORKER_NAMES if getattr(config, name).enable]


def flow_of(config: Config) -> list[list[str]]:
    """この pipeline の配線。recording が種なら routes_list、テキスト起点なら
    text_send_operations が実際に使われる鎖。どちらも鎖には種自身が入らない
    ので、表示用に先頭へ足す。stream_vc は鎖に乗らないので別の 1 本として併記
    する。stream_vc だけの pipeline では text の鎖を描かない — 走りもしない
    tts→playback を配線として見せることになるため。"""
    chains: list[list[str]] = []
    if config.recording.enable:
        chains = [["recording", *chain] for chain in config.recording.routes_list]
    elif enabled_workers(config) != ["stream_vc"]:
        chains = [["(text)", *chain] for chain in config.text_send_operations]
    if config.stream_vc.enable:
        chains.append(list(STREAM_VC_FLOW))
    return chains


def evaluate(config: Config) -> Readiness:
    workers = enabled_workers(config)
    # flow は collect_problems の成否に依存しないので try の外で一度だけ導出する。
    flow = flow_of(config)
    try:
        problems = collect_problems(config)
    except Exception as e:
        # preflight 自体が評価不能 (例: audio extra 未導入でデバイス検査が
        # import 段で失敗)。readiness の失敗で GUI を落とさない (ADR-0045)。
        return Readiness(
            workers=[WorkerReadiness(worker, []) for worker in workers],
            flow=flow,
            error=f"readiness を評価できませんでした: {e}",
        )
    by_worker: dict[str, list[ConfigProblem]] = {worker: [] for worker in workers}
    for problem in problems:
        # enable 済みの worker しか問題を出さないはずだが、取りこぼしを
        # 握り潰さず末尾に見せる。
        by_worker.setdefault(problem.worker, []).append(problem)
    return Readiness(
        workers=[
            WorkerReadiness(worker, problems) for worker, problems in by_worker.items()
        ],
        flow=flow,
    )
