"""Tiny finite-state machines for validating partial JSON values.

Two FSMs live here:

* a JSON-number FSM, used to mask non-numeric tokens during number
  generation;
* a JSON-string FSM, used while generating the content of a string
  parameter (between, but not including, the surrounding quotes).
"""
from typing import Tuple


# --- Number FSM ----------------------------------------------------------

_NUM_START = "start"
_NUM_SIGN = "sign"
_NUM_INT_ZERO = "int_zero"
_NUM_INT_NONZERO = "int_nonzero"
_NUM_DOT = "dot"
_NUM_FRAC = "frac"
_NUM_E = "e_start"
_NUM_E_SIGN = "e_sign"
_NUM_E_INT = "e_int"

_NUMBER_TERMINAL = {
    _NUM_INT_ZERO,
    _NUM_INT_NONZERO,
    _NUM_FRAC,
    _NUM_E_INT,
}


def number_initial_state() -> str:
    """Return the initial state of the JSON-number FSM."""
    return _NUM_START


def number_step(state: str, char: str) -> Tuple[str, bool]:
    """Advance the JSON-number FSM by one character.

    Returns ``(new_state, ok)``. If ``ok`` is ``False`` the character
    cannot extend a JSON number from this state.
    """
    if state == _NUM_START:
        if char == "-":
            return _NUM_SIGN, True
        if char == "0":
            return _NUM_INT_ZERO, True
        if char.isdigit():
            return _NUM_INT_NONZERO, True
        return state, False
    if state == _NUM_SIGN:
        if char == "0":
            return _NUM_INT_ZERO, True
        if char.isdigit():
            return _NUM_INT_NONZERO, True
        return state, False
    if state == _NUM_INT_ZERO:
        if char == ".":
            return _NUM_DOT, True
        if char in ("e", "E"):
            return _NUM_E, True
        return state, False
    if state == _NUM_INT_NONZERO:
        if char.isdigit():
            return _NUM_INT_NONZERO, True
        if char == ".":
            return _NUM_DOT, True
        if char in ("e", "E"):
            return _NUM_E, True
        return state, False
    if state == _NUM_DOT:
        if char.isdigit():
            return _NUM_FRAC, True
        return state, False
    if state == _NUM_FRAC:
        if char.isdigit():
            return _NUM_FRAC, True
        if char in ("e", "E"):
            return _NUM_E, True
        return state, False
    if state == _NUM_E:
        if char in ("+", "-"):
            return _NUM_E_SIGN, True
        if char.isdigit():
            return _NUM_E_INT, True
        return state, False
    if state == _NUM_E_SIGN:
        if char.isdigit():
            return _NUM_E_INT, True
        return state, False
    if state == _NUM_E_INT:
        if char.isdigit():
            return _NUM_E_INT, True
        return state, False
    return state, False


def number_consume(state: str, text: str) -> Tuple[str, bool]:
    """Try to consume every character of ``text``.

    Returns ``(final_state, ok)``. ``ok`` is ``False`` as soon as one
    character is rejected.
    """
    s = state
    for c in text:
        s, ok = number_step(s, c)
        if not ok:
            return s, False
    return s, True


def number_can_terminate(state: str) -> bool:
    """Return ``True`` if the FSM is in a state where the number is complete."""
    return state in _NUMBER_TERMINAL


# --- String FSM ----------------------------------------------------------

_STR_NORMAL = "normal"
_STR_ESCAPE = "escape"
_STR_DONE = "done"


def string_initial_state() -> str:
    """Return the initial state of the JSON-string FSM (already inside the string)."""
    return _STR_NORMAL


def string_step(state: str, char: str) -> Tuple[str, bool]:
    """Advance the JSON-string FSM by one character.

    Reaching ``done`` means the closing unescaped ``"`` was just consumed.
    """
    if state == _STR_DONE:
        return state, False
    if state == _STR_NORMAL:
        if char == '"':
            return _STR_DONE, True
        if char == "\\":
            return _STR_ESCAPE, True
        if ord(char) < 0x20:
            return state, False
        return _STR_NORMAL, True
    if state == _STR_ESCAPE:
        if char in ('"', "\\", "/", "b", "f", "n", "r", "t", "u"):
            return _STR_NORMAL, True
        return state, False
    return state, False


def string_done(state: str) -> bool:
    """Return ``True`` if the string FSM has consumed the closing quote."""
    return state == _STR_DONE
