"""Dataset generation for hard examples.

Chains the corruption operators in :mod:`schemaforge.hardexamples.operators`
over clean seeds from :mod:`schemaforge.hardexamples.seeds` and emits
:class:`schemaforge.evaluation.harness.EvalRecord` objects whose tags name the
applied operators plus a ``severity=<value>`` tag, so results can be sliced per
operator AND per severity exactly as SCHEMAFORGE_V2_RESEARCH_DIRECTION.md §6
requires.

Determinism is mandatory: the same ``seed`` produces byte-identical JSONL
output.  Every source of randomness flows from the single seeded
``random.Random`` (one per-example ``random.Random`` is derived from it via
``getrandbits``), no module-level ``random`` calls are made, and output is
serialized with ``sort_keys=True`` so no ordering assumption leaks in.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from schemaforge.evaluation.harness import EvalRecord
from schemaforge.hardexamples.operators import OPERATORS
from schemaforge.hardexamples.seeds import build_seed
from schemaforge.registry import all_schemas, get_schema, held_out_schemas, training_schemas


def apply_operators(
    text: str,
    gold: dict,
    spec,
    ops: list[str],
    rng: random.Random,
    severity: float,
) -> tuple[str, dict, list[str]]:
    """Chain ``ops`` (operator names) over ``text``/``gold`` in order.

    ``severity`` is applied to every operator in the chain.  Returns
    ``(text, gold, tags)`` with ``tags`` the accumulated operator tags in
    application order.  The final gold is validated against ``spec.model``;
    seeds validate and ``nest`` is the only gold-changing operator, so an
    invalid gold here indicates a pipeline bug and raises ``ValueError``.
    """
    tags: list[str] = []
    for name in ops:
        try:
            operator = OPERATORS[name]
        except KeyError:
            raise ValueError(
                f"unknown operator {name!r}; known: {', '.join(sorted(OPERATORS))}"
            ) from None
        text, gold, new_tags = operator(text, gold, rng, severity)
        tags.extend(new_tags)
    try:
        spec.model.model_validate(gold)
    except Exception as exc:  # noqa: BLE001 - re-raised as a pipeline invariant error
        raise ValueError(f"gold no longer validates against schema {spec.name!r}: {exc}") from exc
    return text, gold, tags


def generate_dataset(
    schema_names: list[str] | str,
    n_per_schema: int,
    severities: list[float],
    operator_mix: list[str],
    seed: int,
    *,
    split: str = "any",
) -> list[EvalRecord]:
    """Build a deterministic list of :class:`EvalRecord`.

    For each schema name (sorted) and each severity (in the given order),
    ``n_per_schema`` examples are generated from clean seeds via
    :func:`build_seed` and corrupted by ``operator_mix``.  An unknown schema
    name raises ``KeyError``.  Every record's tags name the applied operators
    plus ``severity=<value>``; an empty ``operator_mix`` emits uncorrupted
    seeds tagged ``clean``.

    ``split`` guards the held-out-schema discipline (charter §12): the three
    held-out schemas (``insurance_claim``, ``conversation``, ``kg_triple``)
    must never leak into a training dataset.

    - ``"train"`` raises ``ValueError`` naming the offending schemas when any
      requested schema is held out; an empty ``schema_names`` defaults to all
      training (non-held-out) schemas.
    - ``"eval"`` allows anything; an empty ``schema_names`` defaults to exactly
      the held-out schemas.
    - ``"any"`` keeps the current behaviour; an empty ``schema_names`` defaults
      to every registered schema.
    """
    if split not in ("train", "eval", "any"):
        raise ValueError(f"split must be 'train', 'eval' or 'any'; got {split!r}")
    if isinstance(schema_names, str):
        schema_names = [part.strip() for part in schema_names.split(",") if part.strip()]

    if split == "train":
        held_out_names = {spec.name for spec in held_out_schemas()}
        offenders = sorted(set(schema_names) & held_out_names) if schema_names else []
        if offenders:
            raise ValueError(
                f"split='train' must not include held-out schemas, but requested: {', '.join(offenders)}"
            )
        if not schema_names:
            schema_names = [spec.name for spec in training_schemas()]
    elif split == "eval":
        if not schema_names:
            schema_names = [spec.name for spec in held_out_schemas()]
    elif not schema_names:
        schema_names = [spec.name for spec in all_schemas()]

    names = sorted(schema_names)
    for name in names:
        get_schema(name)  # raises KeyError for unknown schema names

    master = random.Random(seed)
    records: list[EvalRecord] = []
    for name in names:
        spec = get_schema(name)
        for severity in severities:
            for _ in range(n_per_schema):
                rng = random.Random(master.getrandbits(32))
                text, gold = build_seed(name, rng)
                if operator_mix:
                    text, gold, tags = apply_operators(text, gold, spec, operator_mix, rng, severity)
                    tags = tags + [f"severity={severity}"]
                else:
                    tags = ["clean", "severity=0.0"]
                records.append(EvalRecord(schema=name, source_text=text, reference=gold, tags=tags))
    return records


def serialize_records(records: list[EvalRecord]) -> str:
    """Return the JSONL text for ``records``, deterministically.

    Keys at every nesting level are sorted (``sort_keys=True``) so the output
    does not depend on dict insertion order.
    """
    lines = []
    for record in records:
        payload = {
            "schema": record.schema,
            "source_text": record.source_text,
            "reference": record.reference,
            "tags": record.tags,
        }
        lines.append(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def _split_list_arg(tokens: list[str] | None) -> list[str]:
    """Flatten ``nargs='+'`` tokens, splitting each on commas.

    Accepts both ``a b c`` and ``a,b,c`` (and mixed ``a, b``) forms, dropping
    empty and whitespace-only tokens.
    """
    if not tokens:
        return []
    return [part.strip() for token in tokens for part in token.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: write a JSONL hard-example dataset to ``--out``."""
    parser = argparse.ArgumentParser(
        description="Generate a deterministic JSONL hard-example dataset for SchemaForge V2."
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=[],
        help="schema names, comma- and/or space-separated (default: every registered schema)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="examples per schema per severity",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        default=["0.0,0.5,1.0"],
        help="severity levels, comma- and/or space-separated",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=[],
        help="operator names, comma- and/or space-separated (default: all ten)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "eval", "any"),
        default="any",
        help="dataset split: 'train' rejects held-out schemas, 'eval' allows anything "
        "(defaults to the held-out schemas), 'any' allows everything",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="emit uncorrupted seeds (severity 0) for the baseline slice",
    )
    args = parser.parse_args(argv)

    schemas = _split_list_arg(args.schemas)
    if len(schemas) == 1 and schemas[0].casefold() == "all":
        schemas_arg = ""
    else:
        schemas_arg = schemas

    severities = [float(part) for part in _split_list_arg(args.severities)]
    operators = _split_list_arg(args.operators) or list(OPERATORS)

    if args.clean:
        records = generate_dataset(schemas_arg, args.n, [0.0], [], args.seed, split=args.split)
    else:
        records = generate_dataset(schemas_arg, args.n, severities, operators, args.seed, split=args.split)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(serialize_records(records))
    print(f"wrote {len(records)} records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
