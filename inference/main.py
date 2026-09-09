from __future__ import annotations

import argparse
from pathlib import Path

from extract import EXTRACTORS, extract_run
from spectral import SIGN_MODES, spectral_run

ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    p = argparse.ArgumentParser(description="CoT hidden state extraction + spectral embedding")

    # 데이터 선택
    p.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    p.add_argument("--level", required=True,
                   help="env_name 그대로. 예: BabyAI-GoToObj-v0, "
                        "CustomBabyAI-GoToRedBall-Small-4Dists-v0")
    p.add_argument("--status", default="success", choices=["success", "failure"],
                   help="spectral 단계에서 읽을 대상 (extract는 항상 둘 다 저장)")
    p.add_argument("--methods", nargs="+", default=["full_sequence"],
                   choices=list(EXTRACTORS))
    p.add_argument("--mode", default="no_thinking", choices=["no_thinking", "thinking"],
                   help="읽을 파일: data-dir/{task}_{mode}.jsonl (generate/cot_*.py 출력명 규칙)")

    # 경로
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    p.add_argument("--spectral-dir", type=Path, default=ROOT / "latent" / "spectral_states")

    # 단계 제어
    p.add_argument("--no-extract", dest="extract", action="store_false",
                   help="hidden state 재사용, 스펙트럴만 재계산")
    p.add_argument("--no-spectral", dest="spectral", action="store_false",
                   help="추출만 하고 종료")

    # 스펙트럴 설정
    p.add_argument("-k", type=int, nargs="+", default=[8])
    p.add_argument("--scale", type=str, nargs="+", default=["true"], choices=["true", "false"])
    p.add_argument("--sign-mode", nargs="+", default=["data"], choices=list(SIGN_MODES),
                   help="고유벡터 부호 보정 방식. 여러 개 주면 각각 따로 저장됨 "
                        "(spectral_states/.../k8_scaled_sign-<mode>)")

    a = p.parse_args()
    a.scale = [s == "true" for s in a.scale]
    return a


def main():
    a = parse_args()

    for method in a.methods:
        if a.extract:
            extract_run(data_dir=a.data_dir, out_root=a.hidden_dir,
                        task=a.task, level=a.level, method=method, mode=a.mode)

        if not a.spectral:
            continue

        for k in a.k:
            for scale in a.scale:
                for sign_mode in a.sign_mode:
                    spectral_run(data_root=a.hidden_dir, out_root=a.spectral_dir,
                                 task=a.task, level=a.level, method=method,
                                 status=a.status, k=k, scale=scale, sign_mode=sign_mode)


if __name__ == "__main__":
    main()