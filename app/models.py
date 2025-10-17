from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QAPair:
    question: str
    answer: str
    hint_image: Optional[bytes] = None


@dataclass
class UserSession:
    text: str = ""
    num_questions: int = 0
    qa: List[QAPair] = field(default_factory=list)
    idx: int = 0
    correct: int = 0
    wrong: int = 0
    stage: str = "IDLE"  # IDLE | WAIT_NUM | ASKING | WAIT_ANSWER | CHOOSE_ACTION


SESSIONS: Dict[int, UserSession] = {}


def get_session(user_id: int) -> UserSession:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = UserSession()
    return SESSIONS[user_id]
