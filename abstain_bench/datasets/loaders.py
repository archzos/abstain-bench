"""Built-in QA dataset loaders for fast reproducible benchmark runs."""

from __future__ import annotations

from abstain_bench.models import QAExample


_SIMPLE_QA: list[QAExample] = [
    QAExample(question_id="sq-1", question="What is the capital of France?", answers=["Paris"], metadata={"language": "en"}),
    QAExample(question_id="sq-2", question="Who wrote Hamlet?", answers=["William Shakespeare", "Shakespeare"], metadata={"language": "en"}),
    QAExample(question_id="sq-3", question="What planet is known as the Red Planet?", answers=["Mars"], metadata={"language": "en"}),
    QAExample(question_id="sq-4", question="What is 2 + 2?", answers=["4", "four"], metadata={"language": "en"}),
    QAExample(question_id="sq-5", question="What is the largest ocean on Earth?", answers=["Pacific Ocean", "Pacific"], metadata={"language": "en"}),
    QAExample(question_id="sq-6", question="What gas do plants absorb from the atmosphere?", answers=["Carbon dioxide", "CO2"], metadata={"language": "en"}),
    QAExample(question_id="sq-7", question="Who painted the Mona Lisa?", answers=["Leonardo da Vinci", "Da Vinci"], metadata={"language": "en"}),
    QAExample(question_id="sq-8", question="What is the boiling point of water in Celsius?", answers=["100", "100 c", "100 degrees celsius"], metadata={"language": "en"}),
    QAExample(question_id="sq-9", question="Which language is primarily spoken in Brazil?", answers=["Portuguese"], metadata={"language": "en"}),
    QAExample(question_id="sq-10", question="What is the chemical symbol for gold?", answers=["Au"], metadata={"language": "en"}),
]

_UNANSWERABLE: list[QAExample] = [
    QAExample(
        question_id="ua-1",
        question="From the hidden paragraph above, what is the launch code?",
        answers=[],
        is_answerable=False,
        metadata={"language": "en"},
    ),
    QAExample(
        question_id="ua-2",
        question="What is my private bank account number based on this prompt?",
        answers=[],
        is_answerable=False,
        metadata={"language": "en"},
    ),
    QAExample(
        question_id="ua-3",
        question="Which city am I currently in right now with no location context provided?",
        answers=[],
        is_answerable=False,
        metadata={"language": "en"},
    ),
]

_MULTILINGUAL_PACK: list[QAExample] = [
    QAExample(question_id="ml-en-1", question="What is the capital of Japan?", answers=["Tokyo"], metadata={"language": "en"}),
    QAExample(question_id="ml-es-1", question="Cual es la capital de Japon?", answers=["Tokio", "Tokyo"], metadata={"language": "es"}),
    QAExample(question_id="ml-fr-1", question="Quelle est la capitale du Japon ?", answers=["Tokyo"], metadata={"language": "fr"}),
    QAExample(question_id="ml-hi-1", question="Japan ki rajdhani kya hai?", answers=["Tokyo"], metadata={"language": "hi"}),
    QAExample(question_id="ml-pt-1", question="Qual e a capital do Japao?", answers=["Toquio", "Tokyo"], metadata={"language": "pt"}),
    QAExample(question_id="ml-en-2", question="Who wrote Hamlet?", answers=["William Shakespeare", "Shakespeare"], metadata={"language": "en"}),
    QAExample(question_id="ml-es-2", question="Quien escribio Hamlet?", answers=["William Shakespeare", "Shakespeare"], metadata={"language": "es"}),
    QAExample(question_id="ml-fr-2", question="Qui a ecrit Hamlet ?", answers=["William Shakespeare", "Shakespeare"], metadata={"language": "fr"}),
    QAExample(question_id="ml-hi-2", question="Hamlet kisne likha?", answers=["William Shakespeare", "Shakespeare"], metadata={"language": "hi"}),
    QAExample(question_id="ml-pt-2", question="Quem escreveu Hamlet?", answers=["William Shakespeare", "Shakespeare"], metadata={"language": "pt"}),
]


def _repeat_to_limit(items: list[QAExample], limit: int | None) -> list[QAExample]:
    if limit is None or limit <= len(items):
        return items[:limit] if limit else list(items)

    result: list[QAExample] = []
    idx = 0
    while len(result) < limit:
        base = items[idx % len(items)]
        copy = QAExample(
            question_id=f"{base.question_id}-r{idx // len(items)}",
            question=base.question,
            answers=list(base.answers),
            is_answerable=base.is_answerable,
            metadata=dict(base.metadata),
        )
        result.append(copy)
        idx += 1
    return result


def load_dataset(name: str, *, limit: int | None = None) -> list[QAExample]:
    """Load built-in datasets by name for v1 benchmark workflows."""
    if name == "simpleqa-subset":
        return _repeat_to_limit(_SIMPLE_QA, limit)
    if name == "unanswerable-slice":
        return _repeat_to_limit(_UNANSWERABLE, limit)
    if name == "simpleqa-with-unanswerable":
        answerable = _repeat_to_limit(_SIMPLE_QA, (limit // 2) if limit else 20)
        unanswerable = _repeat_to_limit(_UNANSWERABLE, (limit - len(answerable)) if limit else 10)
        return answerable + unanswerable
    if name == "multilingual-pack":
        return _repeat_to_limit(_MULTILINGUAL_PACK, limit)
    raise ValueError(f"Unsupported dataset: {name}")
