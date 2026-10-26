"""Local MCP server (stdio) that exposes the diabetes risk model to agents.

Read-only. By default it only serves the feature schema; add --with-prediction to also
serve get_diabetes_prediction, which delegates to inference.py. Nothing is stored or logged.

    python src/mcp_server.py [--with-prediction]
"""
import contextlib
import sys
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

import feature_schema as fs

_inference = None


def _load_inference():
    global _inference
    if _inference is None:
        # inference.py (and what it imports) prints on import, stdout is the MCP protocol channel
        with contextlib.redirect_stdout(sys.stderr):
            import inference
            inference.load_model()
        _inference = inference
    return _inference


class PredictionResult(BaseModel):
    risk_class: int = Field(description="1 = higher-risk pattern, 0 = lower-risk pattern")
    risk_label: str
    meaning: str = fs.RISK_MEANING
    is_diagnosis: bool = False


def _bounds(name):
    low, high = fs.CATEGORY_RANGES[name]
    return {"ge": low, "le": high}


def get_feature_schema() -> dict[str, Any]:
    """Describe the model inputs (names, allowed values, answer labels) and what the output means."""
    return fs.get_feature_schema()


def get_diabetes_prediction(
    high_bp: Annotated[int, Field(ge=0, le=1, description="1 = told blood pressure is high, 0 = not")],
    high_chol: Annotated[int, Field(ge=0, le=1, description="1 = told cholesterol is high, 0 = not")],
    bmi: Annotated[float, Field(ge=fs.BMI_RANGE[0], le=fs.BMI_RANGE[1], description="BMI in kg/m2")],
    gen_hlth: Annotated[int, Field(**_bounds("GenHlth"), description="General health, 1 excellent to 5 poor")],
    age: Annotated[int, Field(**_bounds("Age"), description="Age group, 1 (18-24) to 13 (80 or older)")],
    income: Annotated[int, Field(**_bounds("Income"), description="Income band, 1 (under $10,000) to 8 ($75,000+)")],
    education: Annotated[int, Field(**_bounds("Education"), description="Education, 1 (none) to 6 (college graduate)")],
) -> PredictionResult:
    """Run the existing risk model on one set of answers. Returns a class, not a diagnosis or probability."""
    inference = _load_inference()
    try:
        risk_class = inference.predict_risk_class({
            "HighBP": high_bp, "HighChol": high_chol, "BMI": bmi, "GenHlth": gen_hlth,
            "Age": age, "Income": income, "Education": education,
        })
    except inference.PredictionError as exc:
        raise RuntimeError(str(exc)) from exc
    return PredictionResult(risk_class=risk_class, risk_label=fs.RISK_LABELS[risk_class])


def create_server(with_prediction=False):
    server = FastMCP(
        "diabetes-risk",
        instructions="Read-only tools for a screening-style diabetes risk pattern. Not a diagnosis.",
    )
    server.tool()(get_feature_schema)
    if with_prediction:
        server.tool()(get_diabetes_prediction)
    return server


if __name__ == "__main__":
    with_prediction = "--with-prediction" in sys.argv[1:]
    if with_prediction:
        # on Windows, importing numpy while the stdio reader thread is blocked on stdin hangs,
        # so the model has to be loaded before the transport starts
        _load_inference()
    create_server(with_prediction).run(transport="stdio")
