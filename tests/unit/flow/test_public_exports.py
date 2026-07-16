def test_flow_package_exports_standard_steps_and_patterns() -> None:
    from agent_runtime_kit.flow import (
        AgentStep,
        DispatchStep,
        FlowService,
        RuntimeScheduleService,
        SchedulerRunBudget,
        SchedulerRunControlView,
        StepService,
        create_standard_next_step_if_applicable,
    )

    assert AgentStep.step_type == "agent_step"
    assert DispatchStep.step_type == "dispatch_step"
    assert FlowService is not None
    assert StepService is not None
    assert RuntimeScheduleService is not None
    assert SchedulerRunBudget(flow_advances=1, step_starts=0).flow_advances == 1
    assert SchedulerRunControlView(mode="unbounded").mode == "unbounded"
    assert callable(create_standard_next_step_if_applicable)
