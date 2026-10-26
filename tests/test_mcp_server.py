import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.exceptions import ToolError

import mcp_server

SERVER_FILE = str(Path(mcp_server.__file__))
HIGH_RISK = {"high_bp": 1, "high_chol": 1, "bmi": 45.0, "gen_hlth": 5, "age": 13, "income": 1, "education": 1}
LOW_RISK = {"high_bp": 0, "high_chol": 0, "bmi": 22.0, "gen_hlth": 1, "age": 3, "income": 8, "education": 6}


def run(coro):
    return asyncio.run(coro)


def tool_names(server):
    return {tool.name for tool in run(server.list_tools())}


def test_default_server_only_serves_the_schema():
    assert tool_names(mcp_server.create_server()) == {"get_feature_schema"}


def test_prediction_tool_is_opt_in_and_nothing_else_is_exposed():
    assert tool_names(mcp_server.create_server(with_prediction=True)) == {
        "get_feature_schema", "get_diabetes_prediction",
    }


def test_prediction_tool_schema_carries_the_model_limits():
    server = mcp_server.create_server(with_prediction=True)
    tool = next(tool for tool in run(server.list_tools()) if tool.name == "get_diabetes_prediction")
    properties = tool.inputSchema["properties"]
    assert set(tool.inputSchema["required"]) == set(properties) == {
        "high_bp", "high_chol", "bmi", "gen_hlth", "age", "income", "education",
    }
    assert (properties["bmi"]["minimum"], properties["bmi"]["maximum"]) == (12, 98)
    assert (properties["income"]["minimum"], properties["income"]["maximum"]) == (1, 8)
    assert (properties["age"]["minimum"], properties["age"]["maximum"]) == (1, 13)


def test_schema_tool_returns_the_shared_schema():
    server = mcp_server.create_server()
    _, structured = run(server.call_tool("get_feature_schema", {}))
    assert [feature["name"] for feature in structured["features"]][:2] == ["HighBP", "HighChol"]
    assert structured["output"]["is_diagnosis"] is False


def test_prediction_goes_through_the_existing_inference_code(monkeypatch):
    import inference

    seen = []
    monkeypatch.setattr(inference, "predict_risk_class", lambda features: seen.append(features) or 1)
    server = mcp_server.create_server(with_prediction=True)
    _, structured = run(server.call_tool("get_diabetes_prediction", LOW_RISK))
    assert structured["risk_class"] == 1 and structured["risk_label"] == "higher-risk pattern"
    assert structured["is_diagnosis"] is False and "probability" not in structured
    assert seen == [{"HighBP": 0, "HighChol": 0, "BMI": 22.0, "GenHlth": 1, "Age": 3, "Income": 8, "Education": 6}]


def test_prediction_matches_the_saved_model():
    server = mcp_server.create_server(with_prediction=True)
    assert run(server.call_tool("get_diabetes_prediction", HIGH_RISK))[1]["risk_class"] == 1
    assert run(server.call_tool("get_diabetes_prediction", LOW_RISK))[1]["risk_class"] == 0


@pytest.mark.parametrize(
    "override",
    [{"bmi": 500.0}, {"bmi": 3.0}, {"income": 9}, {"age": 0}, {"education": 7}, {"gen_hlth": 6}, {"high_bp": 2}],
)
def test_out_of_range_inputs_are_rejected(override):
    server = mcp_server.create_server(with_prediction=True)
    with pytest.raises(ToolError):
        run(server.call_tool("get_diabetes_prediction", {**LOW_RISK, **override}))


def test_missing_inputs_and_unknown_tools_are_rejected():
    server = mcp_server.create_server(with_prediction=True)
    incomplete = {key: value for key, value in LOW_RISK.items() if key != "age"}
    with pytest.raises(ToolError):
        run(server.call_tool("get_diabetes_prediction", incomplete))
    with pytest.raises(ToolError):
        run(server.call_tool("read_file", {"path": "/etc/passwd"}))


def test_unknown_fields_never_reach_the_model(monkeypatch):
    import inference

    seen = []
    monkeypatch.setattr(inference, "predict_risk_class", lambda features: seen.append(features) or 0)
    server = mcp_server.create_server(with_prediction=True)
    run(server.call_tool("get_diabetes_prediction", {**LOW_RISK, "smoker": 1}))
    assert set(seen[0]) == {"HighBP", "HighChol", "BMI", "GenHlth", "Age", "Income", "Education"}


async def _talk_to_server(args, calls):
    params = StdioServerParameters(command=sys.executable, args=[SERVER_FILE, *args])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await asyncio.wait_for(session.initialize(), 120)
        tools = {tool.name for tool in (await session.list_tools()).tools}
        results = [await asyncio.wait_for(session.call_tool(name, arguments), 120) for name, arguments in calls]
        return tools, results


def test_real_stdio_server_serves_the_schema():
    tools, (schema,) = run(_talk_to_server([], [("get_feature_schema", {})]))
    assert tools == {"get_feature_schema"}
    assert not schema.isError and schema.structuredContent["output"]["is_diagnosis"] is False


def test_real_stdio_server_with_prediction():
    tools, (high, low, bad) = run(_talk_to_server(
        ["--with-prediction"],
        [("get_diabetes_prediction", HIGH_RISK), ("get_diabetes_prediction", LOW_RISK),
         ("get_diabetes_prediction", {**LOW_RISK, "bmi": 500.0})],
    ))
    assert tools == {"get_feature_schema", "get_diabetes_prediction"}
    assert high.structuredContent["risk_class"] == 1
    assert low.structuredContent["risk_class"] == 0
    assert bad.isError
