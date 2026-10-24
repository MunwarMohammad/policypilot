from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filing(Record):
    ticker: str
    cik: int
    company: str
    accession: str
    filing_date: str
    report_date: str
    source_url: str


class Passage(Filing):
    chunk_id: str
    text: str
    chunk_index: int


class Hit(Record):
    passage: Passage
    score: float


class Claim(Record):
    text: str = Field(min_length=1, max_length=2000)
    source_id: str
    quote: str = Field(min_length=1, max_length=2000)


class Answer(Record):
    claims: list[Claim] = Field(default_factory=list, max_length=8)
    insufficient_evidence: bool = False


class Plan(Record):
    query: str = Field(min_length=1, max_length=1000)


class Judgment(Record):
    supported: list[bool]
