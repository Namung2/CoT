## 모두 Level 에서 요구(평가)되는 각 항목들
- Room Navigation (**ROOM**): navigate a 6x6 room.
    - 6x6 방 안에서 길찾기.
- Ignoring Distracting Boxes (**DISTR-BOX**): navigate the environment even when there are multiple distracting grey box objects in it.
    - 방해가 되는 회색 상자 여러 개가 있어도 무시하고 길찾기.
- Ignoring Distractors (**DISTR**): same as DISTR-BOX, but distractor objects can be boxes, keys or balls of any color.
    - DISTR-BOX와 동일하되, 방해물이 상자/열쇠/공 등 임의 색상일 수 있음.
- Maze Navigation (**MAZE**): navigate a 3x3 maze of 6x6 rooms, randomly inter-connected
by doors.
    - 문으로 무작위 연결된 6x6 방 3x3 미로에서 길찾기.
- Unblocking the Way (**UNBLOCK**): navigate the environment even when it requires moving objects out of the way.
    - 길을 막고 있는 물체를 치워야만 이동 가능한 경우에도 길찾기.
- Unlocking Doors (**UNLOCK**): to be able to find the key and unlock the door if the instruction requires this explicitly.
    - 지시어가 명시적으로 요구할 경우, 열쇠를 찾아 문을 잠금 해제할 수 있어야 함.
- Guessing to Unlock Doors (**IMP-UNLOCK**): to solve levels that require unlocking a
door, even if this is not explicitly stated in the instruction.
    - 지시어에 명시돼 있지 않아도, 문을 잠금 해제해야 풀리는 레벨을 해결할 수 있어야 함.
- Go To Instructions (**GOTO**): understand “go to” instructions, 
    - e.g. “go to the red ball”.
    - "가라(go to)" 지시어 이해. 예: "빨간 공으로 가라".
- Open Instructions (**OPEN**): understand “open” instructions, 
    - e.g. “open the door on your left”.
    - "열어라(open)" 지시어 이해. 예: "왼쪽에 있는 문을 열어라".
- Pickup Instructions (**PICKUP**): understand “pick up” instructions, 
    - e.g. “pick up a box”.
    - "집어라(pick up)" 지시어 이해. 예: "상자를 집어라".
- Put Instructions (**PUT**): understand “put” instructions,
    - e.g. “put a ball next to the blue key”.
    - "놓아라(put)" 지시어 이해. 예: "공을 파란 열쇠 옆에 놓아라".
- Location Language (**LOC**): understand instructions where objects are referred to by relative location as well as their shape and color, 
    - e.g. “go to the red ball in front of you”.
    - 물체를 모양/색뿐 아니라 상대적 위치로도 지칭하는 지시어 이해. 예: "네 앞에 있는 빨간 공으로 가라".
- Sequences of Commands (**SEQ**): understand composite instructions requiring an agent to execute a sequence of instruction clauses,
    - e.g. “put red ball next to the green box after you open the door”.
    - 여러 지시절을 순서대로 수행해야 하는 복합 지시어 이해. 예: "문을 연 다음, 빨간 공을 초록 상자 옆에 놓아라".

## 위 competencies들을 수행하는 단위(subgoal) <bot 자체적으로 판단>
- **Open**: Open a door that is in front of the agent.
    - 에이전트 정면에 있는 문을 연다.
- **Close**: Close a door that is in front of the agent.
    - 에이전트 정면에 있는 문을 닫는다.
- **Pickup**: Execute the pickup action (pick up an object).
    - pickup 행동을 실행한다 (물체를 집는다).
- **Drop**: Execute the drop action (drop an object being carried).
    - drop 행동을 실행한다 (들고 있던 물체를 내려놓는다).
- **GoNextTo**: Go next to an object matching a given (type, color) description or next to a cell at a given position.
    - 주어진 (종류, 색) 서술과 일치하는 물체 옆, 또는 주어진 위치의 칸 옆으로 이동한다.
- **Explore**: Uncover previously unseen parts of the environment.
    - 아직 보지 못한 환경의 영역을 탐색해 드러낸다.
