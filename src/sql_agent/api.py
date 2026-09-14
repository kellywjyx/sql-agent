import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import SQLAgent


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    database: str = "demo"
    schema_mode: Literal["auto", "full", "hybrid"] = "auto"
    model_profile: Literal["default", "qwen_v5", "arctic_sql", "xiyan_sql"] = "default"
    generation_strategy: Literal["single", "adaptive_two"] = "single"
    value_mode: Literal["probe", "profiled"] = "probe"
    pipeline_profile: Literal["v5", "v7_grounded", "v11_adapter"] = "v5"


def create_app(agent=None):
    app = FastAPI(title="Read-only SQL Agent")
    root = Path(os.getenv("PORTFOLIO_ARTIFACTS", "artifacts"))
    app.state.agent = agent or SQLAgent(root / "sql-agent/demo.sqlite", artifacts=root)

    @app.get("/health")
    def health():
        try:
            if not app.state.agent.database.exists():
                raise RuntimeError("Run sql-agent seed first")
            details = app.state.agent.client.check()
            if app.state.agent.schema_retriever is not None:
                from .schema import inspect_schema
                schema = inspect_schema(app.state.agent.database)
                details.update(app.state.agent.schema_retriever.check(app.state.agent.database, schema))
                if app.state.agent.value_index is not None:
                    try:
                        details.update(app.state.agent.value_index.check(app.state.agent.database, schema))
                    except RuntimeError:
                        details["v7_grounding"] = "not_ready"
                        details["v7_setup"] = "sql-agent build-value-index --database <path> --artifacts <path>"
            return {"status": "ready", **details}
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/query")
    def query(body: QueryRequest):
        if body.database != "demo":
            raise HTTPException(400, "Only the registered demo database is exposed by this API")
        try:
            result = app.state.agent.ask(body.question, schema_mode=body.schema_mode,
                                         revised_linking=True, semantic_review=False,
                                         model_profile=body.model_profile,
                                         generation_strategy=body.generation_strategy,
                                         value_mode=body.value_mode,
                                         pipeline_profile=body.pipeline_profile,
                                         evidence_mode="generated")
            result["display_truncated"] = len(result["rows"]) > 100
            result["rows"] = result["rows"][:100]
            result.pop("output", None)
            for attempt in result.get("attempts", []):
                attempt.pop("raw_model_response", None)
            return result
        except (RuntimeError, OSError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc
    return app
