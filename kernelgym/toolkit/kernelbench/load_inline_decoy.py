"""Static decoy detection for load_inline (MusaCoder) submissions.

A load_inline submission is a *decoy* when it JIT-compiles a custom extension via
``torch.utils.cpp_extension.load_inline(...)`` but ``ModelNew`` never actually
calls it — i.e. ``forward`` quietly falls back to PyTorch compute ops (or returns
a shortcut) while a real-looking kernel sits compiled-but-unused. Such a sample
can be numerically *correct* yet earns credit without doing the custom work the
prompt requires, which is the reward-hacking case correctness alone cannot catch.

This is a conservative, purely-static (AST) check designed to MINIMIZE false
positives: a submission is flagged only when a load_inline extension is compiled
AND there is no reference to that extension (its variable, a ``self`` attribute
bound to it, or a module-level helper that uses it) anywhere outside the
``load_inline`` assignment itself. Any genuine use clears the flag. It recognizes
both the named-variable form (``ext = load_inline(...)``) and the direct
self-attribute form (``self.ext = load_inline(...)``), and ``load_inline`` import
aliases.
"""

from __future__ import annotations

import ast
from typing import Any, Dict


def _load_inline_names(tree: ast.AST) -> set[str]:
    """Names that resolve to load_inline (the bare name plus any import alias)."""
    names = {"load_inline"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "load_inline" and alias.asname:
                    names.add(alias.asname)
    return names


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def _is_load_inline_call(call: ast.Call, names: set[str]) -> bool:
    name = _callee_name(call)
    return bool(name) and name.split(".")[-1] in names


def detect_load_inline_decoy(model_code: str, *, entry_point: str = "ModelNew") -> Dict[str, Any]:
    """Return a verdict dict: {decoy, reason, extension_vars, extension_functions, used}.

    ``decoy=False`` on any parse error (compile/correctness will surface the real
    problem) and whenever the extension is referenced outside its own
    ``load_inline(...)`` assignment / storage binding. Only flags when confident.
    """
    result: Dict[str, Any] = {
        "decoy": False,
        "reason": "",
        "extension_vars": [],
        "extension_functions": [],
        "used": True,
    }
    try:
        tree = ast.parse(model_code or "")
    except SyntaxError:
        result["reason"] = "unparseable; deferred to compile stage"
        return result

    names = _load_inline_names(tree)
    ext_vars: set[str] = set()        # `ext = load_inline(...)`
    ext_funcs: set[str] = set()       # functions=[...]
    self_ext_attrs: set[str] = set()  # `self.ext = load_inline(...)` or `self.ext = <ext_var>`
    any_load_inline_call = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_load_inline_call(node, names):
            any_load_inline_call = True
            for kw in node.keywords:
                if kw.arg == "functions" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            ext_funcs.add(elt.value)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _is_load_inline_call(node.value, names):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    ext_vars.add(target.id)
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    # direct `self.ext = load_inline(...)`
                    self_ext_attrs.add(target.attr)

    # `self.x = <ext_var>` storage bindings. The RHS Load of ext_var is storage,
    # not a compute use, so its node id is excluded from the reference scan.
    self_binding_value_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in ext_vars:
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    self_ext_attrs.add(target.attr)
                    self_binding_value_ids.add(id(node.value))

    has_extension = bool(ext_vars or ext_funcs or self_ext_attrs)
    if not any_load_inline_call and not has_extension:
        # No custom extension compiled at all -> nothing but torch ops / shortcut.
        result.update(
            decoy=True,
            used=False,
            reason="no load_inline(...) call: no custom CUDA extension is compiled",
        )
        return result
    if not ext_vars and not self_ext_attrs:
        # A load_inline call exists but its handle is not statically bound to a
        # name/self-attr we can track (e.g. functional `load_inline(...).fn(...)`).
        # Stay conservative and do not flag.
        result["extension_functions"] = sorted(ext_funcs)
        result["reason"] = "load_inline call present but handle not statically bound; not flagged"
        return result

    # Use scan: a Load of an extension var, or a Load of `self.<attr>` bound to an
    # extension, that is not the storage binding itself.
    used = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in ext_vars
            and id(node) not in self_binding_value_ids
        ):
            used = True
            break
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in self_ext_attrs
        ):
            used = True
            break

    result["extension_vars"] = sorted(ext_vars)
    result["extension_functions"] = sorted(ext_funcs)
    result["used"] = used
    if not used:
        ref = ", ".join(sorted(ext_vars) or sorted(f"self.{a}" for a in self_ext_attrs))
        result["decoy"] = True
        result["reason"] = (
            f"compiled a load_inline extension ({ref}) but ModelNew never references "
            "it; forward likely falls back to torch ops"
        )
    return result
