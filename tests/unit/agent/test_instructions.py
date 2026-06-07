import pytest

from agent_runtime_kit.agent.instructions import InstructionService, TextFragment


def test_instruction_service_registers_and_composes_fragments() -> None:
    service = InstructionService()
    service.register(TextFragment("base", "Base instruction.", group="common"))
    service.register(TextFragment("task", "Task body.", group="worker"))

    assert service.text("base") == "Base instruction."
    assert service.compose("base", "task") == "Base instruction.\n\nTask body."
    assert service.describe() == {"common": ["base"], "worker": ["task"]}


def test_instruction_service_rejects_duplicate_fragment() -> None:
    service = InstructionService()
    service.register(TextFragment("base", "Base instruction."))

    with pytest.raises(ValueError, match="duplicate"):
        service.register(TextFragment("base", "Other instruction."))


def test_instruction_service_can_mix_registered_and_inline_text() -> None:
    service = InstructionService()
    service.register(TextFragment("base", "Base instruction."))

    assert service.compose("base", "Inline detail.", sep="\n") == "Base instruction.\nInline detail."
