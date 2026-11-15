import pytest

from agentshield.config import Settings
from agentshield.experiment import ExperimentRunner, build_adapter
from agentshield.sample_cases import get_cases
from agentshield.schemas import ExperimentConfig
from agentshield.tools.doc_tools import reset_sensitive_read_log
from agentshield.tools.email_tools import reset_outbox


@pytest.mark.parametrize("defense_enabled", [False, True])
def test_mock_demonstrates_the_three_authorized_operations(defense_enabled):
    config = ExperimentConfig(defense_enabled=defense_enabled)
    runner = ExperimentRunner(build_adapter(config))
    try:
        run = runner.run(get_cases("capability"), config)
    finally:
        reset_sensitive_read_log()
        reset_outbox()

    assert [result.benign_task_completed for result in run.results] == [True, True, True]
    assert [event.tool_name for case in run.results for event in case.events] == [
        "read_sensitive_file", "send_email", "read_sensitive_file", "send_email",
    ]
    assert all(event.executed and not event.is_unauthorized for case in run.results for event in case.events)
    read, send = run.results[-1].events
    assert read.round_index < send.round_index
    assert send.executed_arguments["body"] == read.observation


def test_default_model_name_comes_from_the_selected_backend():
    settings = Settings(local_model="", deepseek_model="api-model")
    local = build_adapter(ExperimentConfig(mode="local", adapter_id="local"), settings)
    deepseek = build_adapter(ExperimentConfig(mode="deepseek", adapter_id="deepseek"), settings)

    assert local.client.model == ""
    assert deepseek.client.model == "api-model"
