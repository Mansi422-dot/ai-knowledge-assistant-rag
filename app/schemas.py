from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=3,
        max_length=2000,
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
    )


class Source(BaseModel):
    document: str
    chunk: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
