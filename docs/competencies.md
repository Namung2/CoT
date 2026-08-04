## 모두 Level 에서 요구(평가)되는 각 항목들
- Room Navigation (**ROOM**): navigate a 6x6 room.
- Ignoring Distracting Boxes (**DISTR-BOX**): navigate the environment even when there are multiple distracting grey box objects in it.
- Ignoring Distractors (**DISTR**): same as DISTR-BOX, but distractor objects can be boxes, keys or balls of any color.
- Maze Navigation (**MAZE**): navigate a 3x3 maze of 6x6 rooms, randomly inter-connected
by doors.
- Unblocking the Way (**UNBLOCK**): navigate the environment even when it requires moving objects out of the way.
- Unlocking Doors (**UNLOCK**): to be able to find the key and unlock the door if the instruction requires this explicitly.
- Guessing to Unlock Doors (**IMP-UNLOCK**): to solve levels that require unlocking a
door, even if this is not explicitly stated in the instruction.
- Go To Instructions (**GOTO**): understand “go to” instructions, 
    - e.g. “go to the red ball”.
- Open Instructions (**OPEN**): understand “open” instructions, 
    - e.g. “open the door on your left”.
- Pickup Instructions (**PICKUP**): understand “pick up” instructions, 
    - e.g. “pick up a box”.
- Put Instructions (**PUT**): understand “put” instructions,
    - e.g. “put a ball next to the blue key”.
- Location Language (**LOC**): understand instructions where objects are referred to by relative location as well as their shape and color, 
    - e.g. “go to the red ball in front of you”.
- Sequences of Commands (**SEQ**): understand composite instructions requiring an agent to execute a sequence of instruction clauses,
    - e.g. “put red ball next to the green box after you open the door”.

## 위 competencies들을 수행하는 단위(subgoal) <bot 자체적으로 판단>
- **Open**: Open a door that is in front of the agent.
- **Close**: Close a door that is in front of the agent.
- **Pickup**: Execute the pickup action (pick up an object).
- **Drop**: Execute the drop action (drop an object being carried).
- **GoNextTo**: Go next to an object matching a given (type, color) description or next to a cell at a given position.
- **Explore**: Uncover previously unseen parts of the environment.