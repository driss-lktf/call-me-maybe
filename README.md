*This project has been created as part of the 42 curriculum by zdrii.*

# Call Me Maybe — Function calling for LLMs

## Description

Call Me Maybe translates natural-language prompts into structured JSON
function calls using a small language model (Qwen3-0.6B by default) and
**constrained decoding**. Rather than asking the model to emit JSON and
hoping for the best, we drive its decoding token by token so the result
is always syntactically valid JSON and conforms to the requested
function schema.

Given a prompt like `"What is the sum of 2 and 3?"` and a list of
function definitions, the program produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

The function selection is taken from the model's logits; only the
*structure* is enforced from the outside.

## Instructions

### Install

```sh
make install
```

This runs `uv sync`, which creates a virtual environment, installs
`numpy` and `pydantic`, and installs the local `llm_sdk` package
(it pulls in `torch`, `transformers` and `huggingface-hub`).

### Run

```sh
make run
```

The first run downloads Qwen3-0.6B from Hugging Face (~1.2 GB) into
`~/.cache/huggingface/hub/`. Subsequent runs reuse the cache.

By default the program reads:

* `data/input/functions_definition.json`
* `data/input/function_calling_tests.json`

and writes:

* `data/output/function_calling_results.json`

You can override these paths:

```sh
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Debug

```sh
make debug
```

Runs the entry point under `pdb`.

### Lint

```sh
make lint        # flake8 + mypy with the project flags
make lint-strict # flake8 + mypy --strict (optional)
```

### Clean

```sh
make clean
```

## Algorithm

### High-level pipeline

For every input prompt we:

1. Build a Qwen-style chat prompt that lists the available functions
   in the system message and contains the user request:

   ```
   <|im_start|>system
   ... function descriptions ...
   <|im_end|>
   <|im_start|>user
   <prompt>
   <|im_end|>
   <|im_start|>assistant
   <think>

   </think>

   ```

   The empty `<think></think>` block disables Qwen3's reasoning mode so
   the model jumps straight to producing the JSON answer.

2. Force-emit the structural prefix `{"name": "` (these tokens are
   pushed to the context without consulting the model).

3. **Let the LLM choose the function name token by token.** At each
   step we compute the set of token IDs that, when appended, keep the
   produced string a prefix of at least one candidate name. We then
   pick the highest-logit token in that set. The *selection itself* is
   driven entirely by the raw logits — no heuristic or string match
   on the prompt is involved.

4. Force-emit `", "parameters": {`.

5. For each parameter, in declaration order, force-emit `"<name>": `
   then generate the value with a type-specific FSM:

   * **`string`** — opens a JSON string and lets the model produce
     content freely until it emits the closing quote. Tokens that
     would inject a control character or a malformed escape are
     masked out.
   * **`number`** / **`integer`** — masks any token that is not a
     valid extension of a JSON number; stops as soon as the model's
     preferred token is a terminator (`,`, `}`, whitespace) and the
     FSM is in a complete state.
   * **`boolean`** — same constrained selection as the function name,
     restricted to `true` and `false`.

6. Force-emit `, ` between parameters and `}}` at the end.

7. Validate the result with Pydantic and serialise to JSON.

### Why constrained decoding?

A 0.6 B-parameter model is not reliable at producing valid JSON on its
own. By restricting the candidate tokens at each step we trade a sliver
of expressivity for **100 % structural validity**. Every output we
emit can be parsed by `json.loads` and matches the declared schema by
construction.

## Design decisions

* **Public-API only.** The decoder uses `Small_LLM_Model.encode`,
  `decode` and `get_logits_from_input_ids`. No private SDK attribute
  is touched.
* **Pydantic at both ends.** Function definitions, test prompts and
  produced function calls are all validated by Pydantic models defined
  in `src/schemas.py`.
* **Single LLM load per run.** `FunctionCallingPipeline` builds the
  model once and reuses it for every prompt. The vocabulary (one
  decoded string per token ID) is built lazily on the first
  constrained step and cached in memory.
* **Per-prompt error isolation.** Any exception while processing one
  prompt is caught, logged on stderr and recorded as an `error` entry
  in the output array; the rest of the batch keeps running.
* **No prompt-level reliance on JSON discipline.** The system prompt
  describes the contract for the model's benefit, but the decoder
  does not trust it: correctness is guaranteed by the constraints, not
  by the prompt.

## Performance analysis

On an Apple-silicon Mac using the MPS backend, a typical run looks
like:

* Model loading and first forward pass: ~10 s.
* Vocabulary decode (~152 k tokens, decoded one at a time): ~60–90 s
  on the first pipeline call. This dominates the start-up time.
* Per-prompt constrained generation: 1–5 s, mostly determined by the
  number of value tokens to produce.
* The full eleven-prompt benchmark fits comfortably under the 5-minute
  budget required by the spec.

Because the decoder enforces validity, the only way to "fail" is for
the FSM to become stuck. In practice that does not happen on the
provided prompts: every output is parseable JSON, and parameters land
in the correct types declared in `functions_definition.json`.

## Challenges faced

* **Tokenizer quirks.** Qwen3 uses byte-level BPE, and many tokens
  carry leading whitespace (e.g. `" fn"`). Working at the
  *character* level when checking continuation candidates avoids
  false negatives — we accept any token whose decoded string keeps us
  on a valid prefix, regardless of whether it splits cleanly on
  natural boundaries.
* **Knowing when to stop a number.** We let the model decide: at every
  step we compute the best valid number-extension token *and* the best
  terminator token. If the FSM is in a complete state and the
  terminator is preferred, we stop without consuming it.
* **Disabling Qwen3's reasoning mode.** Without an explicit
  `<think>\n\n</think>` block the model would produce a long
  monologue before answering, forcing the constrained prefix to start
  inside reasoning text. Including the empty think block in the prompt
  skips it.
* **mypy + an external SDK.** The graded `mypy` flags refuse modules
  with no annotated `__init__.py`, so we tell mypy to ignore errors
  inside `llm_sdk` via a `[[tool.mypy.overrides]]` block while still
  type-checking our own code.

## Testing strategy

The project is validated in three ways:

1. **End-to-end run** on the provided test file
   `data/input/function_calling_tests.json`, with manual inspection of
   the produced `data/output/function_calling_results.json`. The output
   file is re-validated by `json.loads` to confirm structural validity.
2. **Schema validation via Pydantic.** Every produced record passes
   through `FunctionCall.model_validate` before being written.
3. **Unit checks of the FSMs** (`src/grammar.py`) by direct calls in a
   Python REPL during development — for instance `number_consume("3")`
   should accept the string and reach a terminal state.

To re-validate manually after a run:

```sh
uv run python -c \
    'import json; print(json.load(open("data/output/function_calling_results.json")))'
```

## Example usage

Input file `data/input/function_calling_tests.json`:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

Output file `data/output/function_calling_results.json` (illustrative):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

## Resources

### Documentation and references

* Hugging Face documentation — *constrained generation* and
  `LogitsProcessor` API.
* The Outlines project (`dottxt-ai/outlines`) — survey of structured
  generation techniques (used as a reference, **not** as a
  dependency: the spec forbids it).
* The Qwen3 model card and chat-template description on Hugging Face
  Hub.
* RFC 8259 — *The JavaScript Object Notation (JSON) Data Interchange
  Format*, used as the source of truth for the number and string
  grammars implemented in `src/grammar.py`.

### Use of AI

AI assistance (Claude Opus, Anthropic) was used during the design and
implementation of the project for the following tasks:

* Discussing the trade-offs of token-aligned versus character-aligned
  candidate matching during constrained decoding.
* Drafting a first version of the JSON-number FSM in
  `src/grammar.py`, then refining its terminating-state semantics.
* Cross-checking the project's allow-list (no `dspy`, no
  `transformers` outside the SDK, no private SDK attributes) and the
  shape of the required Makefile targets.

Every AI-suggested snippet was read, edited and tested before being
kept; nothing was committed without understanding why it is correct.
