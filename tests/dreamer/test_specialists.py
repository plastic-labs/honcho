from src.dreamer.specialists import DeductionSpecialist, InductionSpecialist


class TestDeductionSpecialistControl:
    def test_tool_list_includes_finish_consolidation(self):
        specialist = DeductionSpecialist()
        tool_names = [tool["name"] for tool in specialist.get_tools(peer_card_enabled=True)]

        assert "finish_consolidation" in tool_names
        assert "update_peer_card" in tool_names

    def test_prompt_forces_early_commit_or_finish(self):
        specialist = DeductionSpecialist()
        prompt = specialist.build_system_prompt("user-default-hermes-agent", peer_card_enabled=True)

        assert "Stop discovery after at most 3 tool calls" in prompt
        assert "Call `update_peer_card`" in prompt
        assert "call `finish_consolidation`" in prompt


class TestInductionSpecialistControl:
    def test_tool_list_includes_finish_consolidation(self):
        specialist = InductionSpecialist()
        tool_names = [tool["name"] for tool in specialist.get_tools(peer_card_enabled=True)]

        assert "finish_consolidation" in tool_names
        assert "update_peer_card" in tool_names

    def test_prompt_forces_early_commit_or_finish(self):
        specialist = InductionSpecialist()
        prompt = specialist.build_system_prompt("user-default-hermes-agent", peer_card_enabled=True)

        assert "Stop discovery after at most 3 tool calls" in prompt
        assert "Call `update_peer_card`" in prompt
        assert "call `finish_consolidation`" in prompt
