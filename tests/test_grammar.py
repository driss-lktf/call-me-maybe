"""Unit tests for the JSON grammar FSMs in src/grammar.py."""
import pytest

from src.grammar import (
    number_can_terminate,
    number_consume,
    number_initial_state,
    number_step,
    string_done,
    string_initial_state,
    string_step,
)


# ---------------------------------------------------------------------------
# Number FSM
# ---------------------------------------------------------------------------

class TestNumberFSM:
    """Tests for the JSON-number finite-state machine."""

    def test_initial_state_not_terminal(self) -> None:
        assert not number_can_terminate(number_initial_state())

    @pytest.mark.parametrize("digit", list("123456789"))
    def test_single_digit_terminal(self, digit: str) -> None:
        state = number_initial_state()
        state, ok = number_step(state, digit)
        assert ok
        assert number_can_terminate(state)

    def test_zero_is_terminal(self) -> None:
        state = number_initial_state()
        state, ok = number_step(state, "0")
        assert ok
        assert number_can_terminate(state)

    def test_negative_number(self) -> None:
        state, ok = number_consume(number_initial_state(), "-42")
        assert ok
        assert number_can_terminate(state)

    def test_decimal(self) -> None:
        state, ok = number_consume(number_initial_state(), "3.14")
        assert ok
        assert number_can_terminate(state)

    def test_exponent(self) -> None:
        state, ok = number_consume(number_initial_state(), "1e10")
        assert ok
        assert number_can_terminate(state)

    def test_exponent_negative(self) -> None:
        state, ok = number_consume(number_initial_state(), "2.5E-3")
        assert ok
        assert number_can_terminate(state)

    def test_invalid_leading_dot(self) -> None:
        state, ok = number_consume(number_initial_state(), ".5")
        assert not ok

    def test_invalid_double_minus(self) -> None:
        state, ok = number_consume(number_initial_state(), "--1")
        assert not ok

    def test_partial_decimal_not_terminal(self) -> None:
        state, ok = number_consume(number_initial_state(), "3.")
        assert ok
        assert not number_can_terminate(state)

    def test_empty_string_not_ok(self) -> None:
        state, ok = number_consume(number_initial_state(), "")
        # Empty consume leaves state unchanged (start), still not terminal.
        assert not number_can_terminate(state)

    def test_large_integer(self) -> None:
        state, ok = number_consume(number_initial_state(), "123456789")
        assert ok
        assert number_can_terminate(state)

    def test_zero_decimal(self) -> None:
        state, ok = number_consume(number_initial_state(), "0.0")
        assert ok
        assert number_can_terminate(state)

    def test_reject_alpha(self) -> None:
        state, ok = number_consume(number_initial_state(), "abc")
        assert not ok


# ---------------------------------------------------------------------------
# String FSM
# ---------------------------------------------------------------------------

class TestStringFSM:
    """Tests for the JSON-string finite-state machine."""

    def test_initial_state_not_done(self) -> None:
        assert not string_done(string_initial_state())

    def test_closing_quote_marks_done(self) -> None:
        state = string_initial_state()
        state, ok = string_step(state, '"')
        assert ok
        assert string_done(state)

    def test_plain_char_accepted(self) -> None:
        state = string_initial_state()
        state, ok = string_step(state, "a")
        assert ok
        assert not string_done(state)

    def test_backslash_starts_escape(self) -> None:
        state = string_initial_state()
        state, ok = string_step(state, "\\")
        assert ok
        # Must be followed by a valid escape character.
        state2, ok2 = string_step(state, "n")
        assert ok2
        assert not string_done(state2)

    def test_invalid_escape(self) -> None:
        state = string_initial_state()
        state, _ = string_step(state, "\\")
        _, ok = string_step(state, "x")  # 'x' is not a valid escape
        assert not ok

    def test_control_char_rejected(self) -> None:
        state = string_initial_state()
        _, ok = string_step(state, "\n")
        assert not ok

    def test_tab_rejected(self) -> None:
        state = string_initial_state()
        _, ok = string_step(state, "\t")
        assert not ok

    def test_done_state_rejects_everything(self) -> None:
        state = string_initial_state()
        state, _ = string_step(state, '"')
        assert string_done(state)
        _, ok = string_step(state, "a")
        assert not ok

    def test_unicode_escape(self) -> None:
        state = string_initial_state()
        state, ok = string_step(state, "\\")
        assert ok
        state, ok = string_step(state, "u")
        assert ok

    def test_full_string(self) -> None:
        state = string_initial_state()
        for ch in "hello":
            state, ok = string_step(state, ch)
            assert ok
        state, ok = string_step(state, '"')
        assert ok
        assert string_done(state)
