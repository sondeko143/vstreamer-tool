"""起動前 readiness の評価 (ADR-0045)。

「何が必須か」の権威は vspeech.preflight (ADR-0038) にあり、この module は
それを呼んで表示用に整形するだけ。必須項目の知識をここへ複製しないこと —
複製すると GUI が緑なのに起動して落ちる (逆も) という形で drift が出る。
"""

from dataclasses import dataclass

from vspeech.config import Config
from vspeech.exceptions import ConfigProblem
from vspeech.preflight import collect_problems

# config に <name>.enable を持つ worker。preflight のチェッカーと対象を揃える。
WORKER_NAMES: tuple[str, ...] = (
    "recording",
    "transcription",
    "translation",
    "tts",
    "vc",
    "playback",
    "subtitle",
)


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
    ので、表示用に先頭へ足す。"""
    if config.recording.enable:
        return [["recording", *chain] for chain in config.recording.routes_list]
    return [["(text)", *chain] for chain in config.text_send_operations]


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
