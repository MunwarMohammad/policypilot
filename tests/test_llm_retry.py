from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from groq import BadRequestError

from policypilot.llm import JSONModel
from policypilot.models import Plan


def test_groq_json_rejection_retries_with_schema_validation():
    model = object.__new__(JSONModel)
    model.backend, model.model, model.client = "local", "test", Mock()
    error = BadRequestError(
        "invalid JSON",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.com")),
        body={"code": "json_validate_failed"},
    )
    create = model.client.chat.completions.create
    create.side_effect = [
        error,
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"query":"business segments"}'))
            ]
        ),
    ]
    assert model.generate("Plan search", {}, Plan).query == "business segments"
    assert "response_format" not in create.call_args.kwargs
