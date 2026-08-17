# CoT Latent State Probing — 실험 세팅

> **연구 질문**: LLM의 hidden state $e_t$가 chain-of-thought 도중 latent task structure를
> 표면 토큰 이상으로 인코딩하는가.
>
> **범위**: CTRLS(arXiv 2507.08182)에서 가져오는 것은 **$e_t$ 추출까지**.
> CTRLS를 구현하거나 개선하지 않는다. 클러스터링·전이행렬·RL은 범위 밖.
>
> **BabyAI 선택 이유**: CTRLS는 GSM8K/MATH로 평가하는데 외부 ground-truth latent state
> 라벨이 없다. BabyAI는 매 스텝 `(pos, dir, carrying)`을 환경에서 직접 읽을 수 있고
> 전이가 결정론적이다.

---

## 0. 실험이 필요 없는 것

데이터를 뽑아도 안 바뀐다. 설계의 **전제**로 사용한다.

| ID | 사실 | 근거 |
|---|---|---|
| **M1** | $G_t = E_t^\top E_t = \sum_i h_i h_i^\top$ → **토큰 순서 완전 불변.** step 내부 구성성은 $e_t$에 들어가지 않음 | CTRLS Eq.(5) 정의 |
| **M2** | $\mathrm{tr}(G_t) = \sum_i \lVert h_i\rVert^2$ → 정규화 없으면 $e_t$ 크기가 **토큰 수를 인코딩** | 동일 |
| **M3** | causal LM은 prefix 불변 → $G_t^{\text{Yu}} = E_A[t]^\top E_A[t]$, 그 trace는 $t$에 **단조증가 보장** | Yu et al. 2509.00190 ($G_t = G_{t-1} + \tilde G_t$) |
| **M4** | Eq.(5)는 well-defined하지 않음. $q \leftrightarrow -q$, 고유값 축퇴 시 부분공간 회전. **well-defined한 건 $\lambda$와 사영행렬 $P_k$뿐** | Davis–Kahan |
| **M5** | `PutNextInstr`는 `ActionInstr` 서브클래스 → PutNextLocal 계열에서 `instr_done ≡ 0` (죽은 라벨) | minigrid 소스 + 실행 확인 |
| **M6** | `Box.toggle`이 박스를 `contains`로 치환 → 미션 대상 소멸 가능. PutNextLocal에는 문이 없음 | minigrid `world_object.py` |
| **M7** | `generate()`는 `position_ids = attention_mask.cumsum(-1)-1`로 교정하지만, plain forward는 `arange`를 씀 → **두 경로가 다름** | [`generation/utils.py:762-765`](https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/generation/utils.py) vs [`models/qwen2/modeling_qwen2.py:363`](https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen2/modeling_qwen2.py) (v5.15.0) |
| **M8** | 샘플링은 전역 RNG(`torch.multinomial(probs, ...)`, `probs.shape=(batch,vocab)`) → RNG 소비가 배치 구성에 의존 | `generation/utils.py:2923` |

**M3이 가장 중요.** Yu et al.의 핵심 결과(클러스터가 step 위치와 $\rho{=}1.0$)는 인코더 구성에서
따라나오는 정리이지 LLM에 대한 발견이 아니다. 따라서 **method A(누적 prefix)는 분석 대상이 아니라
position artifact 양성대조군**이다.

**M7/M8은 batch=1로 고정하면 발생하지 않는다** (§5 참조). 기록만 남긴다.

---

## 1. 가설

### A. 인코더 자체

CTRLS에서 가져갈 게 $e_t$뿐인데, $e_t$의 스펙트럼 구조가 실질적으로 rank-1이면 가져갈 게 없다.
데이터 예산을 쓰기 전에 알아야 한다.

| ID | 가설 | 검정량 | 기각 조건 |  
|---|---|---|---|
| **A1** | $K_t = E_tE_t^\top$ 가 사실상 rank-1이라 $k{=}8$이 무의미 | mean off-diagonal cosine, effective rank $\dfrac{(\sum\lambda)^2}{\sum\lambda^2}$, $\lambda_1/\sum\lambda$ | eff. rank $\gg 1$ |
| **A3** | eigengap이 작으면 $q_i$가 불안정 (Davis–Kahan) | no-op 변환 전후 성분별 $\cos(q_i, q_i')$, 부호 뒤집힘률 | 모든 $i \le k$ 안정 |
| **A4** | *(가설 아님 — 측정)* 파이프라인 노이즈 바닥 | T1, T6 | — |

- $K_t$와 $G_t$는 **0이 아닌 고유값이 완전히 동일** ($E = USV^\top$ → $G = VS^2V^\top$, $K = US^2U^\top$).
  토큰 레벨로 가서 새로 얻는 것은 $U$(토큰별 loading)뿐이며, **순서 정보가 남아 있는 유일한 곳**이다.
- off-diag cosine이 높으면 $e_t \approx \sqrt{\lambda_1}q_1 \approx$ (공통 방향)×(크기).
  즉 **A1의 숫자 자체가 "Eq.(5)가 사실상 평균 방향 하나"임을 판정**한다.
- ~~A2 (mean/last/max-pool 대비)~~ — CTRLS에 없는 경쟁 embedding 정의라 현 단계 제외.

### B. $e_t$가 무엇을 담는가 — lookup 문제

**문제 정의.** 주장 H1은 "$e_t$가 표면 토큰에 없는 state 정보를 담는다"인데,
경쟁 가설 H0 "$e_t$의 정보는 전부 $x_t$ 문자열 안에 이미 있었다"가 모든 것을 설명하면
H1이 반증 불가능해진다. c3(템플릿 텍스트 35종)에서 $x_t \to e_t$가 문자 그대로 표였던 것이 극단이고,
LLM 자유 텍스트로 바꿔도 형태만 바뀐다 — CoT가 `"I can see the red key ahead and I'm holding nothing"`
이면 그 문장은 $s_t$의 **서술**이므로 $e_t$가 $s_t$를 맞히는 것은 당연하다.

같은 $(x_t, e_t, s_t)$ 위에 다섯 예측기를 올린다.

| 기호 | 무엇 |
|---|---|
| $P_{\text{chance}}$ | 다수 클래스 비율 |
| $P_{\text{pos}}$ | $t$만으로 $s_t$ 예측 |
| $P_{\text{surf}}$ | $x_t$ **텍스트만** (TF-IDF / frozen encoder, LLM 미사용) |
| $P_{\text{shuf}}$ | 같은 $x_t$ + **다른 에피소드 prefix**로 재추출한 $e_t$ |
| $P_e$ | $e_t$ |

| ID | 가설 | 검정량 | 기각 조건 |
|---|---|---|---|
| **B1** | $e_t$ = 표면 텍스트 (**lookup**) | $\Delta_{\text{surf}} = P_e - P_{\text{surf}}$ | $> 0$ |
| **B2** | history 미통합 | $\Delta_{\text{hist}} = P_e - P_{\text{shuf}}$ | $> 0$ |
| **B3** | $e_t$가 담는 건 $t$뿐 | $\Delta_{\text{pos}} = P_e - P_{\text{pos}}$ | $> 0$ |
| **B4** | $\lVert e_t\rVert \propto n_t$ | $\mathrm{corr}(\lVert e_t\rVert, n_t)$, scale on/off | 상관 소멸 |

- **응집도 검정(같은 $s_t$에 여러 CoT)은 하지 않는다.** CTRLS의 $z_t$는 실제 생성된 그 궤적의 상태이고,
  CTRLS는 $e_t$를 재샘플링으로 검증하지 않는다 (Appendix F의 다중 샘플링은 정답 궤적 확보 +
  §6.2 pass@20용). 위 네 통제는 **새 생성 없이** 계산되므로 결정론적 디코딩과 충돌하지 않는다.
- **B2 보너스**: terminal step을 모든 에피소드 동일 상수로 두면 표면 텍스트가 완벽히 통제되므로,
  에피소드 간 $\cos(e_T^{(i)}, e_T^{(j)})$ 분포가 $\Delta_{\text{hist}}$를 직접 준다.

### C. 프롬프트 / 포맷

| ID | 가설 | 대조 | 진단 지표 | 근거 |
|---|---|---|---|---|
| **C1** | 좌표를 주면 CoT에 복사되어 pos probe가 자명해짐 | P1 vs P2 | coordinate mention rate, $\Delta_{\text{surf}}$(pos 한정) | 자체. LLM-BabyBench는 omniscient+좌표 사용 |
| **C2** | few-shot exemplar가 문장 패턴 복제로 lookup 재발 | P1 vs P4 | unique/total, 4-gram 반복률, 파싱 성공률 | ReAct 2210.03629: PaLM-8B/62B에서 in-context로 reasoning+acting 동시 학습이 어려워 ReAct 프롬프팅이 최악 |
| **C3** | 분절 granularity가 $e_t$ 성질을 바꿈 | **공짜 축** | $\cos(e_t, e_{t+1})$, $T$ 분포 | TRACES 2604.21057 |

**~~C4 (마커) 삭제.**~~ `Step k:`를 프롬프트로 강제하는 것은 CTRLS에 없다.
Appendix F는 Wei et al. 2022만 인용하고, Wei의 rationale은 자유 산문이다.
Figure 6의 번호는 논문의 **렌더링**이며, Yu et al.의 마커는 **사후 분절**이다.
TRACES는 특수 토큰 생성을 프롬프트로 강제하는 방식이 신뢰성·효율이 낮고 모델이 이 하위 과제로
사전학습되지 않아 오류를 유발한다고 명시한다.
→ 마커를 안 쓰면 **"숫자 $k$가 텍스트에 $t$를 새긴다"는 위험이 통째로 사라진다.**

---

## 2. 공통 세팅

### 환경

```
level     BabyAI-PutNextLocalS6N4-v0
actions   left / right / forward / pickup / drop
seed      필터 없음. reward > 0 만
```

**레벨 근거** — bot 롤아웃 실측 (seed 300~1000):

| 레벨 | T med | T mean | (pos,dir) 재방문 | carrying 전환 | room |
|---|---|---|---|---|---|
| GoToLocalS5N2 | 3 | 2.9 | 0.0% | 0.00 | 5 |
| GoToLocal | 4 | 5.3 | 0.2% | 0.01 | 8 |
| GoToObj | 5 | 5.0 | 0.0% | 0.00 | 8 |
| PickupLoc | 6 | 6.1 | 0.2% | 0.00 | 8 |
| PutNextLocalS5N3 | 7 | 7.0 | 0.0% | 1.00 | 5 |
| **PutNextLocalS6N4** | **8** | **8.4** | 0.0% | **1.01** | 6 |
| PutNextLocal | 12 | 12.3 | 0.1% | 1.01 | 8 |

- GoTo/Pickup 계열은 **carrying 전환이 0.00** — latent state가 순수 위치뿐이라
  $\Delta_{\text{pos}}$로 걷어내려는 것만 남는다. (PickupLoc은 pickup이 종료 행동이라 전환이 궤적 밖)
- $T \le 6$이면 lag가 5개 이하라 시간축 분석이 성립하지 않는다.
- S6N4가 S5N3보다 $T$ 중앙값 8 vs 6으로 우세. 난이도 차이는 미미(물체 3→4, 내부 3×3→4×4).
- 재방문율 0%는 **bot이 최적이라서**이며, LLM 궤적의 재방문율은 파일럿에서 별도 측정.
- 사다리: 성공률 보고 S5N3 하향 / S8N8 상향.
- bot이 못 푸는 레벨(`PutNextS*Carrying`, `KeyInBox`) 아님.

**액션 근거**
- `toggle` 제외 — M6. PutNextLocal에 문이 없고 `Box.toggle`이 대상을 소멸시킨다.
  bot 궤적 2815스텝에서 `toggle`/`done` 출현 **0회**.

**seed 필터 근거** — BabyAI 공식 `scripts/make_agent_demos.py`:
```python
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps")
...
if reward > 0 and (args.filter_steps == 0 or len(images) <= args.filter_steps):
    demos.append(...)
```
**상한만 있고 하한 없음, 기본 무필터.** 실패 시 seed를 넘긴다.
→ `min_steps`는 레퍼런스에 없어 **철회**. 가시성 필터도 **철회**(§공변량 참조).

### 라벨 chance level (bot 궤적 2815 steps)

| 라벨 | classes | majority |
|---|---|---|
| `dir` | 4 | 25.4% |
| `pos` | 9 | 21.6% |
| `action` | 5 | 31.8% |
| `carrying` | 2 | 55.9% |
| `front_obj` | 5 | 61.7% (empty) |
| ~~`instr_done`~~ | 1 | **100% — 죽은 라벨 (M5)** |

`action` 분포: forward 896 / right 703 / left 416 / pickup 400 / drop 400

### 생성

```
mode        one-shot (중간 관측 없음)
model       Qwen/Qwen2.5-3B-Instruct    (생성 = 인코딩 동일 모델)
batch_size  1                            (필수 — §5)
prompt      mission + 관측만. subgoal·계획·정답경로 금지
            "Let's think step by step."
            액션은 말미 <actions> 블록 1회
            추론 텍스트 안 액션명 금지
분절        사후. \n\n 기본
step 0      mission 제외 (CoT만 분절)
terminal    성공 시에만 하드코딩 **상수** 1개 append
```

**mode 근거**
- CTRLS Assumption 5.2($Q_\phi(z_t|x_{\le t})$)가 문자 그대로 성립 — 하나의 자기회귀 생성.
- step-wise는 $o_t$가 바로 앞 prefix에 있어 "$e_t$가 $s_t$를 담는가"가 **자명해진다.**
- 비용: 에피소드당 LLM 호출 1회.
- ReAct는 thought를 행동공간에 넣는($\hat A = A \cup L$) 상호작용 프로토콜이라 다른 범주.
  중간 관측이 없는 우리 세팅엔 dense/sparse 축이 정의되지 않는다 → **최후 보루로 보류**.

**프롬프트 근거**
- CTRLS Appendix F: *"We follow the prompt design in (Wei et al., 2022) to guide step-wise CoT generation."*
  → 프롬프트는 **문제 진술만**. 해법 구조도 중간 상태 라벨도 없다.
- 마커 강제 없음 — C4 삭제 사유 참조.

**분절 근거**
- TRACES 2604.21057: 단락 수준(`\n\n`)이 *"현재 문헌에서 가장 널리 채택된 정의"*.
  Local Causal Attribution(2606.21821), PUMA(2605.17672)도 동일 계보.
- **`step_spans` ↔ `action_spans` 대응을 요구하지 않는다.**
  action은 채점용, step은 $e_t$용으로 완전히 분리.
- 위험: TRACES의 `\n\n` 관찰은 DeepSeek-R1/QwQ/GPT 같은 reasoning 모델 기준이며
  **Qwen2.5-3B-Instruct는 reasoning 모델이 아니다.** → 파일럿 게이트에서 확인.
  폴백: MuCRASP(2605.25842)의 marker taxonomy(구조적 delimiter + 논리 커넥티브),
  그것도 없으면 equal-interval 분할.

**terminal 근거**
- 상수로 두면 표면 텍스트가 완벽히 통제되어 B2의 $\Delta_{\text{hist}}$를 공짜로 측정할 수 있다.
- 좌표 자동 배제.

### 디코딩 — 파일럿이 결정

| | greedy | temperature 0.7 + top-k |
|---|---|---|
| 재현성 | ✅ | ✅ (고정 시드) |
| 반복 퇴화 위험 | **높음** (Holtzman et al. ICLR 2020) | 낮음 |
| 근거 | ReAct: 전 방법 greedy decoding | CTRLS Appendix F("sampling with temperature and top-k"), §6.2 $\eta \in \{0.5, 0.7\}$ |

- **판정 기준**: 파일럿의 에피소드 간 4-gram 반복률.
- 샘플링 선택 시 `torch.manual_seed(env_seed)`를 매 `generate()` 직전 — batch=1이므로 안전.
  이는 레퍼런스 없는 **우리 규약**으로 명시한다.
  (BabyAI `utils.seed()`는 실행 단위 1회, CTRLS·LLM-BabyBench는 시드 규약 없음.)
- top-k 값은 CTRLS 미명시 → **우리 결정으로 보고** (K=50 제안).
- 온도는 0.7 시작, 형식 준수가 깨지면 0.5 (둘 다 CTRLS 범위).

### 채점

```
성공   env reward > 0. 이진.
bot    같은 seed 독립 롤아웃 1회 → bot_reference_len 만
```

- CTRLS Appendix F(정답 궤적만 보존), LLM-BabyBench, ReAct, BabyAI **전부 이진**.
- ~~$\Phi$(잔여 스텝), bot_agree, 이탈점~~ — 레퍼런스 없는 자체 발명이었으므로 **철회**.
- bot 롤아웃의 용도는 Efficiency Ratio(LLM-BabyBench)와 해당 seed의 해결 가능성 확인뿐.
- BabyAI bot은 순수 규칙 기반(BFS + subgoal stack, `torch`/`nn` 참조 0건, 무작위성 0)이지만
  **최적이 아니라 휴리스틱**이므로 `bot_optimal_len`이 아니라 `bot_reference_len`으로 명명.

---

## 3. 실험 격자

### 생성 비용이 드는 3셀

| | 관측 | exemplar | 검정 | 근거 |
|---|---|---|---|---|
| **P1** | partial (GLAM BabyAI-Text describer, 좌표 없음) | 없음 | **본 실험** — A·B 전부 | GLAM 2302.02662 |
| **P2** | full (LLM-BabyBench StructuredFormatter, 절대좌표) | 없음 | **C1** | LLM-BabyBench 2505.12135 |
| ~~P3~~ | ~~마커 없음~~ | | **삭제** | C4 근거 붕괴 |
| **P4** | partial | 3-shot | **C2** | ReAct의 ALFWorld 어노테이션 방식 |

- **P1/P2는 동일 seed 집합.** 전체물체 가시 케이스(~30%)에서는 **정보량이 동일하고
  인코딩만 다르므로**(상대 서술 vs 절대 좌표) C1이 정보량 교란 없이 순수하게 측정된다.
- GLAM의 부분관측은 **상호작용 루프 안**이라는 단서를 명시한다(1.5M step RL 후 결과).
  "GLAM이 partial이니 one-shot partial도 된다"는 논증은 성립하지 않는다.

### 가시성 — 필터가 아니라 공변량

측정 분포 (S6N4, N=600): 대상 2개 41.5% / 1개 34.2% / 0개 24.3%.
전체 물체 가시 ~30%.

```
n_targets_visible    0|1|2    프롬프트에서 복원 가능한 파생 통계
all_objects_visible  bool     환경 특권 정보. 분석 전용, 프롬프트에 절대 미포함
```

**"0개 보이는 군에서 성공률 0"이 나오면 그것이 one-shot partial 성립 여부에 대한 결과다.**
필터로 지우면 그 검정을 못 한다.

### 공짜 축 — P1 데이터 재사용, 새 생성 없음

```
분절   S-nl(\n\n) / S-sent / S-merge / S-marker(폴백)
추출   method B(현재 step) / method A(누적 prefix = M3 양성대조군)
표현   λ-only(Yu) / [√λ·q](CTRLS Eq.5) / projector P_k
스윕   k, scale(√n_t 정규화), fix_sign, layer
통제   P_surf / P_pos / P_shuf / chance
```

**layer는 스윕 대상으로 승격.** CTRLS는 미명시, Yu et al.은 "최종층을 128차원으로 사영"이라고만
하고 사영 방법을 밝히지 않는다. 두 논문이 비운 자리다.

---

## 4. 데이터 스키마

### `{cell}/steps.jsonl`

```
id, level, level_variant, seed, num_objs
mode            "oneshot"
obs_mode        "partial" | "full"
mission
prompt          {system, user}          # 렌더링된 원문 그대로
raw_output      str                     # 파싱 전 모델 출력 전체
token_ids       [int]                   # 실제 forward에 넣은 그것
step_spans      [[s,e]]                 # token_ids 기준. \n\n 분절
action_spans    [[s,e]]                 # 누출 감사용 (e_t span에는 미사용)
steps           [str]                   # token_ids[s:e] 디코드와 일치해야 함
answer          {action_seq, success, reward, final_pos, final_dir}
n_planned, n_bad_pairs, truncated
outcome         "success"|"failed"|"incomplete"|"unparsed"
```

### `{cell}/labels.jsonl` — `id`로 조인, 행 순서 일치

```
id, n_exec, outcome
action, pos, dir, carrying, front_obj          # per-step
n_targets_visible, all_objects_visible          # per-episode 공변량
bot_reference_len
terminal {...}
```

**라벨은 action 단위 그대로 유지한다.** reasoning 단위 라벨을 미리 집계해 저장하면
후보 하나에 고정되므로, 집계는 사후에 한다.

### `{cell}/manifest.json`

```
model, dtype, chat_template_applied
decoding {temperature, top_k, do_sample, seed_policy, max_new_tokens}
tokenizer_name, transformers_version, minigrid_version, torch_version
pad_token_id
level, level_variant, num_objs, seed_range
created
```

### 불변식 (assert)

```
tok.decode(token_ids[s:e]) == steps[i]          # span 정합 — 하드 게이트
step_spans[i][1] <= step_spans[i+1][0]          # 겹침/역전 없음
outcome == "success"  ⟺  answer.success
```

**개수 등식(`len(step_spans) == n_exec`)은 걸지 않는다.**
hidden state 1개 = action 1개가 아니며, 1 step : n action을 허용한다.

성공만 `cases/`에 저장하되, **실패 궤적의 통계(T 분포, unique/total, 반복률)는 별도 기록**한다
(성공 필터가 다양성을 줄이는지 감사용).

---

## 5. batch=1 고정과 그 귀결

one-shot이므로 에피소드당 `generate()` 1회, 추출 1회. **배치가 필요 없다.**

**따라서 다음이 발생하지 않는다** (기록만 남김):
- left-padding의 RoPE 위치 시프트 — M7. 패딩 자체가 없음
- 커널 reduction 순서에 의한 배치 크기 의존 수치 차이
- 샘플링 RNG가 배치 구성에 의존하는 문제 — M8

**남는 노이즈원은 둘뿐이다**:
- **T1**: 생성 시 hidden state(KV 캐시, 쿼리 길이 1의 증분 attention)
  vs teacher-forcing 재투입(전체 attention). 커널 경로가 다르다.
- **T6**: bf16 vs fp32

**T1이 이후 저장 전략을 결정한다.**
통과하면 `token_ids`만 저장하고 나중에 재투입 → layer·$k$ 스윕이 재생성 없이 공짜.
통과하지 않으면 생성 시점에 hidden state를 전부 저장 → layer 변경 시마다 재생성 필요.

### RoPE 참고

$$\langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle$$

절대 위치가 상쇄되고 상대 위치만 남는다 (Su et al. arXiv 2104.09864, Qwen2.5는 `rope_theta = 1e6`).
절대위치 의존이 남는 경로는 attention sink(Xiao et al. arXiv 2309.17453),
$\cos/\sin$ 계산의 bf16 반올림, rope_scaling(Qwen2.5 기본 off) 정도다.

---

## 6. 실행 순서

### STEP 0 — P1 파일럿 1 에피소드
```
batch=1, generate(output_hidden_states=True, return_dict_in_generate=True)
→ 실제 prompt + 실제 생성 텍스트 + token_ids + 생성 시 hidden state 확보
```

### STEP 1 — 그 실시퀀스로 전부 측정

| ID | 무엇 | 왜 |
|---|---|---|
| **1-a** | **인과성**: 뒤쪽 토큰 $j$ 교체 → `H[:, :j]` 전 layer 불변인지 | 결정적 테스트. 마스킹된 위치는 softmax 후 0이라 비트 단위로 같거나 최하위 비트만 다름 |
| **1-b** | prefix 불변: `H_full[:M]` vs `H_pref[:M]` | 1-a의 따름정리. method A/B 구분의 전제 |
| **1-c** | span 정합: `tok.decode(ids[s:e]) == step_text` (`\n\n` 경계) | 깨지면 $E_t$가 옆 step 토큰을 문다. **프롬프트 포맷 수정 필요** |
| **1-d** | **T1**: 생성 시 hidden state vs teacher-forcing 재투입 | **저장 전략 결정** |
| **1-e** | T6: bf16 vs fp32 | 노이즈 바닥 |
| **1-f** | **A1**: off-diag cosine, effective rank, $\lambda_1/\sum\lambda$, eigengap | $k$ 상한 |
| **1-g** | A3: 1-d/1-e 전후 $q_i$ cosine·부호 뒤집힘 | $k$ 상한 실증 |
| **1-h** | trace: `trA` 단조성, `trA/N_t` 상수성, `trB/n_t` | M2/M3 실증 |

**1-a → 1-b → 1-c → 1-d 순.** 1-a가 통과해야 나머지가 의미를 갖는다.

노이즈 바닥은 숫자 하나가 아니라 **분포**로 저장:
```json
noise_floor.json
{ "transform": "T1", "layer": -1, "dtype": "bf16",
  "h_cos": {"p50":..., "p05":..., "p01":..., "min":...},
  "e_cos": {...},
  "lambda_rel_err": [per-i],
  "q_cos": [per-i], "q_sign_flip_rate": [per-i] }
```
**`q_cos[i]`가 무너지는 $i$가 $k$의 실증적 상한이다.**

### STEP 2 — P1 파일럿 n=30

| 게이트 | 결정하는 것 |
|---|---|
| span assert 통과율 | 데이터 유효성 |
| 파싱 성공률 | 낮으면 P4(few-shot) 우선순위 상승 |
| action leakage rate | 높으면 프롬프트 수정 |
| coordinate mention rate | C1의 전제 |
| 4-gram 반복률 (에피소드 간) | **디코딩 확정** |
| `\n\n` 분절 후 step 수 분포 | 분절 성립 여부. 중앙값 1이면 폴백 |
| 가시성별 성공률 | **one-shot partial 성립 여부** |
| LLM 궤적 재방문율 | state 중복 (imagination 유의미성) |
| unique/total, dup% | lookup 조기 경보 |
| dead label 탐지 | 상수 컬럼 자동 검출 |
| $\mathrm{corr}(n_t, t)$ | B4의 전제 |

### STEP 3 — P1 n=150
A·B 그룹 전부 + 공짜 축 전부. A1 실데이터 재측정 → $k$ 최종 확정.

### STEP 4
B 결과를 보고 P2 / P4 중 필요한 것만.

---

## 7. 미확정 항목

| 항목 | 판정 방법 |
|---|---|
| 디코딩 (greedy vs temperature) | STEP 2의 4-gram 반복률 |
| top-k 값 | CTRLS 미명시 → 우리 결정으로 보고 |
| 온도 (0.7 → 0.5) | 형식 준수율 |
| 레벨 변종 (S5N3 / S6N4 / S8N8) | 성공률 |
| exemplar 도입 시점 | 파싱 성공률 |
| layer | STEP 3 스윕 |
| $k$ | STEP 1의 `q_cos[i]` + A1 |
| 분절 방식 | STEP 2의 step 수 분포 |

---

## 참고문헌

| 약칭 | arXiv / 출처 | 용도 |
|---|---|---|
| CTRLS | 2507.08182 | 이론 근거. Eq.(5), Assumption 5.1/5.2, Appendix F |
| Yu et al. | 2509.00190 | CTRLS 자매 논문. 누적 Gram, $\lambda$-only. **M3의 대상** |
| GLAM | 2302.02662 | BabyAI-Text partial observation describer |
| LLM-BabyBench | 2505.12135 | StructuredFormatter, Efficiency Ratio, 이진 성공 |
| BabyAI | ICLR 2019 | 레벨 정의, demo length, bot |
| BabyAI repo | github.com/mila-iqia/babyai | `scripts/make_agent_demos.py` 필터 규약 |
| ReAct | 2210.03629 | greedy decoding, dense/sparse thought, 소형 모델 경고 |
| Wei et al. | 2201.11903 | CoT 프롬프트 원형 (CTRLS가 인용) |
| TRACES | 2604.21057 | step 분절 계보 4종. `\n\n`이 최다 채택 |
| PUMA | 2605.17672 | blank-line + merge 분절 |
| Local Causal Attr. | 2606.21821 | double-newline 분절 |
| MuCRASP | 2605.25842 | marker taxonomy 폴백 |
| RoPE | 2104.09864 | 상대위치 성질 |
| Attention sink | 2309.17453 | 절대위치 의존 후보 |
| Holtzman et al. | 1904.09751 | greedy 반복 퇴화 |
| transformers | v5.15.0 | M7, M8 소스 근거 |