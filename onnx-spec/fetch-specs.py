#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

"""Fetch ONNX operator specs and write per-operator markdown files to ops/.

Usage: ./onnx-spec/fetch-specs.py
"""

import os
from pathlib import Path

import onnx
from onnx import defs

# Only the default ONNX domain
DOMAIN = ""

# Attribute type name mapping
ATTR_TYPE_NAMES = {
    0: "UNDEFINED",
    1: "FLOAT",
    2: "INT",
    3: "STRING",
    4: "TENSOR",
    5: "GRAPH",
    6: "FLOATS",
    7: "INTS",
    8: "STRINGS",
    9: "TENSORS",
    10: "GRAPHS",
    11: "SPARSE_TENSOR",
    12: "SPARSE_TENSORS",
    13: "TYPE_PROTO",
    14: "TYPE_PROTOS",
}

# Input/output option mapping
FORMAL_PARAM_OPTION = {
    0: "Single",
    1: "Optional",
    2: "Variadic",
}


def get_latest_schemas():
    """Get the latest version of each operator schema."""
    all_schemas = defs.get_all_schemas_with_history()
    latest = {}
    for schema in all_schemas:
        if schema.domain != DOMAIN:
            continue
        name = schema.name
        if name not in latest or schema.since_version > latest[name].since_version:
            latest[name] = schema
    return latest


def format_attribute(attr):
    """Format a single attribute as markdown."""
    type_name = ATTR_TYPE_NAMES.get(attr.type, f"UNKNOWN({attr.type})")
    required = "required" if attr.required else "optional"
    line = f"- **{attr.name}** ({type_name}, {required})"
    if attr.description:
        desc = attr.description.strip().split("\n")[0]  # first line only
        line += f": {desc}"
    return line


def format_io(param, direction):
    """Format a single input or output as markdown."""
    option = FORMAL_PARAM_OPTION.get(param.option, "Single")
    type_str = param.type_str if param.type_str else ""
    line = f"- **{param.name}**"
    parts = []
    if type_str:
        parts.append(type_str)
    if option != "Single":
        parts.append(option.lower())
    if parts:
        line += f" ({', '.join(parts)})"
    if param.description:
        desc = param.description.strip().split("\n")[0]
        line += f": {desc}"
    return line


def format_type_constraints(schema):
    """Format type constraints as markdown."""
    lines = []
    for tc in schema.type_constraints:
        types = ", ".join(sorted(tc.allowed_type_strs))
        lines.append(f"- **{tc.type_param_str}**: {types}")
        if tc.description:
            desc = tc.description.strip().split("\n")[0]
            lines.append(f"  {desc}")
    return lines


def schema_to_markdown(schema):
    """Convert an ONNX schema to a markdown string."""
    lines = []
    lines.append(f"# {schema.name}")
    lines.append("")
    lines.append(f"Since opset **{schema.since_version}**")
    lines.append("")

    if schema.doc:
        lines.append("## Description")
        lines.append("")
        lines.append(schema.doc.strip())
        lines.append("")

    # Attributes
    attrs = list(schema.attributes.values())
    if attrs:
        lines.append("## Attributes")
        lines.append("")
        for attr in sorted(attrs, key=lambda a: a.name):
            lines.append(format_attribute(attr))
        lines.append("")

    # Inputs
    if schema.inputs:
        min_inputs = schema.min_input
        max_inputs = schema.max_input
        lines.append(f"## Inputs ({min_inputs} - {max_inputs})")
        lines.append("")
        for inp in schema.inputs:
            lines.append(format_io(inp, "input"))
        lines.append("")

    # Outputs
    if schema.outputs:
        min_outputs = schema.min_output
        max_outputs = schema.max_output
        lines.append(f"## Outputs ({min_outputs} - {max_outputs})")
        lines.append("")
        for out in schema.outputs:
            lines.append(format_io(out, "output"))
        lines.append("")

    # Type constraints
    tc_lines = format_type_constraints(schema)
    if tc_lines:
        lines.append("## Type Constraints")
        lines.append("")
        lines.extend(tc_lines)
        lines.append("")

    return "\n".join(lines)


def main():
    script_dir = Path(__file__).resolve().parent
    ops_dir = script_dir / "ops"
    ops_dir.mkdir(exist_ok=True)

    # Clean existing files
    for f in ops_dir.glob("*.md"):
        f.unlink()

    schemas = get_latest_schemas()
    print(f"Writing {len(schemas)} operator specs to {ops_dir}/")

    for name in sorted(schemas):
        schema = schemas[name]
        md = schema_to_markdown(schema)
        filepath = ops_dir / f"{name}.md"
        filepath.write_text(md)

    print("Done.")


if __name__ == "__main__":
    main()
