from __future__ import annotations

import argparse
from pathlib import Path

from extract import extract_run
from spectral import SIGN_MODES, N_FRONT, N_BACK, spectral_run, parse_fallback

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="CoT hidden state extraction + spectral embedding")

    # 데이터 선택
    p.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    p.add_argument("--level", required=True,
                   help="env_name 그대로. 예: BabyAI-GoToObj-v0, "
                        "CustomBabyAI-GoToRedBall-Small-4Dists-v0")
    p.add_argument("--status", nargs="+", default=["success", "failure"],
                   choices=["success", "failure"],
                   help="spectral 단계에서 읽을 대상 (extract는 항상 둘 다 저장). "
                        "기본값은 둘 다")
    p.add_argument("--mode", default="no_thinking", choices=["no_thinking", "thinking"],
                   help="읽을 파일: data-dir/{task}_{mode}.jsonl "
                        "(generation/scripts/cot_*.py 출력명 규칙)")

    # 경로
    p.add_argument("--data-dir", type=Path, default=ROOT / "generation" / "trajectory")
    p.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    p.add_argument("--spectral-dir", type=Path, default=ROOT / "latent" / "spectral")

    # 단계 제어
    p.add_argument("--no-extract", dest="extract", action="store_false",
                   help="hidden state 재사용, 스펙트럴만 재계산")
    p.add_argument("--no-spectral", dest="spectral", action="store_false",
                   help="추출만 하고 종료")

    # 추출 설정
    p.add_argument("--chunk", type=int, default=256,
                   help="이 개수마다 .pt 파일 하나 (status별로 따로 센다)")
    p.add_argument("--limit", type=int, default=None,
                   help="앞에서 이 개수만 추출. 본 실행 전에 skip 사유 분포와 "
                        "청크 파일 크기를 확인할 때 쓴다")

    # 스펙트럴 설정
    p.add_argument("-k", type=int, nargs="+", default=[8])
    p.add_argument("--scale", type=str, nargs="+", default=["true"], choices=["true", "false"])
    p.add_argument("--sign-mode", nargs="+", default=["data"], choices=list(SIGN_MODES),
                   help="고유벡터 부호 보정 방식. 여러 개 주면 각각 따로 저장됨 "
                        "(spectral/.../k8_scaled_sign-<mode>_f<n_front>_b<n_back>)")
    p.add_argument("--n-front", type=int, default=N_FRONT,
                   help="구간 앞(형식 문구 뒤)에서 저장할 토큰별 e 개수. 0 이면 안 저장")
    p.add_argument("--n-back", type=int, default=N_BACK,
                   help="구간 뒤에서 저장할 토큰별 e 개수. 1 이면 구간 마지막 e_t 만")
    p.add_argument("--fallback-marker", nargs="+", default=[], metavar="TASK=S:T",
                   help="--n-front > 0 인데 원본 jsonl 에서 에피소드를 못 찾을 때 쓸 "
                        "형식 토큰 수 (S=step 헤더, T=터미널 문구, 예: predict=5:7)")

    a = p.parse_args()
    a.scale = [s == "true" for s in a.scale]
    a.status = list(dict.fromkeys(a.status))   # 중복 제거, 순서 유지
    return a


def main():
    a = parse_args()

    if a.extract:
        extract_run(data_dir=a.data_dir, out_root=a.hidden_dir,
                    task=a.task, level=a.level, mode=a.mode,
                    chunk=a.chunk, limit=a.limit)

    if not a.spectral:
        return

    configs = [(k, scale, sm) for k in a.k for scale in a.scale for sm in a.sign_mode]
    fallback = parse_fallback(a.fallback_marker)
    for status in a.status:
        # extract 직후라 항상 다시 만든다 (낡은 청크 파일도 지움)
        stats = spectral_run(a.hidden_dir, a.spectral_dir, a.task, a.level, status, configs,
                             n_front=a.n_front, n_back=a.n_back,
                             traj_dir=a.data_dir, mode=a.mode, fallback=fallback, overwrite=True)
        print(f"{a.task}/{a.level}/{status}: {dict(stats)}")


if __name__ == "__main__":
    main()