"""Static reward-hacking checks for KernelBench submissions.

Python model code and native sources take different paths.  A method name is
not enough to identify framework compute: ``math.sqrt`` and an extension
export are legitimate calls that share names with tensor methods.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Collection, Mapping

from kernelgym.schema.precision import normalize_precision


@dataclass(frozen=True)
class StaticCheckResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    precision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "precision": self.precision,
        }


_COMPUTE = frozenset(
    {
        "matmul",
        "mm",
        "bmm",
        "einsum",
        "conv1d",
        "conv2d",
        "conv3d",
        "softmax",
        "log_softmax",
        "relu",
        "gelu",
        "layer_norm",
        "batch_norm",
        "sum",
        "mean",
        "max",
        "min",
        "prod",
        "cumprod",
        "cumsum",
        "exp",
        "sqrt",
        "rsqrt",
        "norm",
        "cross_entropy",
        "nll_loss",
    }
)
_TORCH_MODULES = frozenset(
    {"ReLU", "GELU", "Softmax", "LogSoftmax", "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}
)
_NATIVE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp")
_MESSAGES = {
    "code_bypass": "Contains try/except fallback bypass pattern",
    "timing_event_patch": "Reassigns timing or CUDA synchronization function",
    "thread_injection": "Uses threading or multiprocessing inside submitted kernel code",
    "lazy_eval": "Defines or constructs lazy/fake tensor objects",
    "stream_injection": "Uses explicit CUDA stream control",
    "framework_compute": "Uses PyTorch/ATen compute instead of custom CUDA kernels",
    "precision_downgrade": "Precision downgrade detected: required FP32 but code uses FP16",
}


def _mask_c_like_noncode(
    code: str,
    *,
    mask_literals: bool = True,
) -> str:
    """Mask C/C++ comments and literals, preserving newlines and positions."""
    chars, index, quote = list(code), 0, None
    while index < len(chars):
        char, following = chars[index], chars[index + 1] if index + 1 < len(chars) else ""
        if quote is not None:
            if char == "\\":
                if mask_literals:
                    chars[index] = " "
                if mask_literals and index + 1 < len(chars) and chars[index + 1] != "\n":
                    chars[index + 1] = " "
                index += 2
                continue
            if mask_literals:
                chars[index] = char if char == "\n" else " "
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "/" and following == "/":
            chars[index] = chars[index + 1] = " "
            index += 2
            while index < len(chars) and chars[index] != "\n":
                chars[index] = " "
                index += 1
            continue
        if char == "/" and following == "*":
            chars[index] = chars[index + 1] = " "
            index += 2
            while index < len(chars):
                if chars[index] == "*" and index + 1 < len(chars) and chars[index + 1] == "/":
                    chars[index] = chars[index + 1] = " "
                    index += 2
                    break
                if chars[index] != "\n":
                    chars[index] = " "
                index += 1
            continue
        if char in ("'", '"'):
            quote = char
            if mask_literals:
                chars[index] = " "
        index += 1
    return "".join(chars)


def mask_native_noncode(code: str) -> str:
    """Mask native comments and literals while retaining line/column offsets."""

    return _mask_c_like_noncode(code)


def mask_native_comments(code: str) -> str:
    """Mask native comments but retain literals, including include paths."""

    return _mask_c_like_noncode(code, mask_literals=False)


class _PythonIssueVisitor(ast.NodeVisitor):
    def __init__(self, allowed_extension_modules: Collection[str]) -> None:
        self.allowed_extensions = frozenset(allowed_extension_modules)
        self.aliases: dict[str, tuple[str, tuple[str, ...]]] = {
            "math": ("math", ()),
            "cmath": ("cmath", ()),
            "time": ("time", ()),
            "torch": ("torch", ()),
        }
        self.blocked_names: set[str] = set()
        self.issues: set[str] = set()
        self.imported_extensions: set[str] = set()
        self.referenced_extensions: set[str] = set()
        self.extension_calls: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root, *rest = alias.name.split(".")
            name = alias.asname or root
            # ``import torch.cuda`` binds ``torch``, whereas
            # ``import torch.cuda as cuda`` binds the full dotted module.
            self.aliases[name] = (root, tuple(rest) if alias.asname else ())
            self.blocked_names.discard(name)
            if alias.name in self.allowed_extensions:
                self.imported_extensions.add(alias.name)
            if root in {"threading", "multiprocessing"} or alias.name.startswith("concurrent.futures"):
                self.issues.add("thread_injection")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            for alias in node.names:
                if alias.name in self.allowed_extensions:
                    name = alias.asname or alias.name
                    self.aliases[name] = (alias.name, ())
                    self.blocked_names.discard(name)
                    self.imported_extensions.add(alias.name)
            self.generic_visit(node)
            return
        root, *rest = node.module.split(".")
        for alias in node.names:
            if alias.name != "*":
                name = alias.asname or alias.name
                self.aliases[name] = (root, tuple(rest + [alias.name]))
                self.blocked_names.discard(name)
        if node.module in self.allowed_extensions:
            self.imported_extensions.add(node.module)
        if root in {"threading", "multiprocessing"} or node.module.startswith("concurrent.futures"):
            self.issues.add("thread_injection")
        self.generic_visit(node)

    def _extension_target(self, root: str, attrs: tuple[str, ...]) -> tuple[str, str] | None:
        if root in self.allowed_extensions and attrs:
            return root, attrs[-1]
        if root == "torch" and len(attrs) >= 3 and attrs[0] == "ops" and attrs[1] in self.allowed_extensions:
            return attrs[1], attrs[2]
        return None

    @staticmethod
    def _is_timing_target(resolved: tuple[str, tuple[str, ...]] | None) -> bool:
        if resolved is None:
            return False
        root, attrs = resolved
        return (root, attrs) in {
            ("torch", ("cuda", "Event", "record")),
            ("torch", ("cuda", "Event", "elapsed_time")),
            ("torch", ("cuda", "synchronize")),
            ("torch", ("cuda", "Event")),
            ("time", ("perf_counter",)),
            ("time", ("time",)),
        }

    def _record_assignment_targets(self, targets: list[ast.expr]) -> None:
        for target in targets:
            if self._is_timing_target(self._resolved(target)):
                self.issues.add("timing_event_patch")

    @staticmethod
    def _bound_names(target: ast.AST) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [name for item in target.elts for name in _PythonIssueVisitor._bound_names(item)]
        return []

    def _block_targets(self, targets: list[ast.AST]) -> None:
        for target in targets:
            for name in self._bound_names(target):
                self.aliases.pop(name, None)
                self.blocked_names.add(name)

    def _resolved(self, node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
        if isinstance(node, ast.Name):
            if node.id in self.blocked_names:
                return None
            return self.aliases.get(node.id, (node.id, ()))
        if isinstance(node, ast.Attribute):
            parent = self._resolved(node.value)
            return (parent[0], parent[1] + (node.attr,)) if parent else None
        if isinstance(node, ast.Call):
            # getattr(torch, "sum")(x) is a literal spelling of a framework
            # call. A non-literal attribute remains intentionally unknown.
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and "getattr" not in self.aliases
                and "getattr" not in self.blocked_names
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                parent = self._resolved(node.args[0])
                return (parent[0], parent[1] + (node.args[1].value,)) if parent else None
            # A generic call returns a new value, not the called function's
            # namespace. Keeping the function provenance here would wrongly
            # classify ``extension.identity(x).half()`` as extension.half.
            return None
        return None

    def _assign_aliases(self, targets: list[ast.expr], value: ast.AST) -> None:
        resolved = self._resolved(value)
        for target in targets:
            for name in self._bound_names(target):
                # Assignment is order-aware: e = torch makes later e.sum a
                # torch call; an unknown rebind invalidates an extension alias.
                if resolved is None:
                    self.aliases.pop(name, None)
                    self.blocked_names.add(name)
                else:
                    self.aliases[name] = resolved
                    self.blocked_names.discard(name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_assignment_targets(node.targets)
        self._assign_aliases(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_assignment_targets([node.target])
            self._assign_aliases([node.target], node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_assignment_targets([node.target])
        if isinstance(node.target, ast.Name):
            self.aliases.pop(node.target.id, None)
            self.blocked_names.add(node.target.id)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._assign_aliases([node.target], node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._record_assignment_targets(node.targets)
        self._block_targets(node.targets)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._block_targets([node.target])
        for child in [*node.body, *node.orelse]:
            self.visit(child)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)  # type: ignore[arg-type]

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._block_targets([item.optional_vars])
        for child in node.body:
            self.visit(child)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)  # type: ignore[arg-type]

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        saved_aliases = self.aliases.copy()
        saved_blocked_names = self.blocked_names.copy()
        if node.name is not None:
            self.aliases.pop(node.name, None)
            self.blocked_names.add(node.name)
        for child in node.body:
            self.visit(child)
        self.aliases = saved_aliases
        self.blocked_names = saved_blocked_names

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp) -> None:
        saved_aliases = self.aliases.copy()
        saved_blocked_names = self.blocked_names.copy()
        for generator in node.generators:
            self.visit(generator.iter)
            self._block_targets([generator.target])
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.aliases = saved_aliases
        self.blocked_names = saved_blocked_names

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_Try(self, node: ast.Try) -> None:
        if node.handlers:
            self.issues.add("code_bypass")
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.AST) -> None:
        if getattr(node, "handlers", []):
            self.issues.add("code_bypass")
        self.generic_visit(node)

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef) -> None:
        saved_aliases = self.aliases.copy()
        saved_blocked_names = self.blocked_names.copy()
        arguments = getattr(node, "args", None)
        if arguments is not None:
            for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
                self.aliases.pop(argument.arg, None)
                self.blocked_names.add(argument.arg)
            if arguments.vararg is not None:
                self.aliases.pop(arguments.vararg.arg, None)
                self.blocked_names.add(arguments.vararg.arg)
            if arguments.kwarg is not None:
                self.aliases.pop(arguments.kwarg.arg, None)
                self.blocked_names.add(arguments.kwarg.arg)
        self.generic_visit(node)
        self.aliases = saved_aliases
        self.blocked_names = saved_blocked_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(
            (resolved := self._resolved(base))
            and resolved[0] == "torch"
            and resolved[1]
            and resolved[1][-1] == "Tensor"
            for base in node.bases
        ):
            self.issues.add("lazy_eval")
        self._visit_scoped(node)

    def visit_Call(self, node: ast.Call) -> None:
        resolved = self._resolved(node.func)
        # The receiver can be intentionally unknown, but an explicit Torch
        # FP16 dtype is still unambiguous. Check it independently so
        # x.to(torch.float16) cannot evade the FP32 gate.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
            and any(
                (item := self._resolved(arg)) and item[0] == "torch" and item[1] and item[1][-1] in {"float16", "half"}
                for arg in [*node.args, *(kw.value for kw in node.keywords)]
            )
        ):
            self.issues.add("precision_downgrade")
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"half", "float16"}:
            extension_target = self._extension_target(*resolved) if resolved is not None else None
            if extension_target is None:
                self.issues.add("precision_downgrade")
        if resolved:
            root, attrs = resolved
            leaf = attrs[-1] if attrs else ""
            extension_target = self._extension_target(root, attrs)
            if extension_target is not None:
                extension_module, extension_call = extension_target
                self.referenced_extensions.add(extension_module)
                self.extension_calls.add(extension_call)
            torch_ops_compute = (
                root == "torch" and len(attrs) >= 3 and attrs[:2] == ("ops", "aten") and attrs[2] in _COMPUTE
            )
            if (
                extension_target is None
                and root == "torch"
                and (leaf in _COMPUTE or leaf in _TORCH_MODULES or torch_ops_compute)
            ):
                self.issues.add("framework_compute")
            if leaf == "astype" and any(
                (item := self._resolved(arg)) and item[0] == "triton" and item[1] and item[1][-1] == "float16"
                for arg in [*node.args, *(kw.value for kw in node.keywords)]
            ):
                self.issues.add("precision_downgrade")
            # Unknown receivers are deliberately left to the fail-closed runtime
            # ATen gate.  This avoids treating unrelated extension/config APIs
            # as tensors solely because they share a method name.
            if extension_target is None and leaf in {"half", "float16"}:
                self.issues.add("precision_downgrade")
            if root == "torch" and (
                (attrs[:1] == ("cuda",) and leaf in {"Stream", "stream"}) or leaf in {"wait_stream", "record_stream"}
            ):
                self.issues.add("stream_injection")
            if leaf == "_make_subclass" or (root == "torch" and attrs[-2:] == ("Tensor", "__new__")):
                self.issues.add("lazy_eval")
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and "setattr" not in self.aliases
            and "setattr" not in self.blocked_names
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            target = self._resolved(node.args[0])
            if target is not None and self._is_timing_target((target[0], target[1] + (node.args[1].value,))):
                self.issues.add("timing_event_patch")
        self.generic_visit(node)


def _python_issues(code: str, allowed_extension_modules: Collection[str]) -> set[str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    visitor = _PythonIssueVisitor(allowed_extension_modules)
    visitor.visit(tree)
    return visitor.issues


def detect_extension_calls(code: str, module_name: str) -> tuple[bool, list[str]]:
    """Resolve calls to one extension module using the same rules as static checks."""

    tree = ast.parse(code)
    visitor = _PythonIssueVisitor({module_name})
    visitor.visit(tree)
    referenced = module_name in visitor.imported_extensions or module_name in visitor.referenced_extensions
    return referenced, sorted(visitor.extension_calls)


def _native_issues(code: str) -> set[str]:
    clean, names, issues = mask_native_noncode(code), "|".join(sorted(_COMPUTE, key=len, reverse=True)), set()
    scope = r"\s*::\s*"
    if re.search(rf"\b(?:at|torch){scope}(?:[A-Za-z_]\w*{scope})*(?:{names})\s*\(", clean):
        issues.add("framework_compute")
    tensor_variables = re.findall(rf"\b(?:torch|at){scope}Tensor\s*(?:[*&]\s*)?([A-Za-z_]\w*)", clean)
    if any(re.search(rf"\b{re.escape(name)}\s*(?:\.|->)\s*(?:{names})\s*\(", clean) for name in tensor_variables):
        issues.add("framework_compute")
    if re.search(
        r"__float2half(_rn)?\s*\(|\(\s*__half\s*\)\s*[\w\->\.]+|static_cast\s*<\s*(__half|half)\s*>\s*\(|\bCUBLAS_COMPUTE_(16F|32F_FAST_16F)\b|\bCUDA_R_16F\b|NumericConverter\s*<\s*half_t\s*,\s*float\s*>|LinearCombination\s*<\s*half_t|type_convert\s*<\s*half_t\s*>\s*\(|tk::half\s*\(",
        clean,
    ):
        issues.add("precision_downgrade")
    if any(re.search(rf"\b{re.escape(name)}\s*(?:\.|->)\s*(?:half|float16)\s*\(", clean) for name in tensor_variables):
        issues.add("precision_downgrade")
    if re.search(r"\b(?:torch|at)::cuda::Stream\s*\(|\btorch::cuda::stream\b", clean):
        issues.add("stream_injection")
    return issues


def _issues(code: str, language: str = "auto", allowed_extension_modules: Collection[str] = ()) -> set[str]:
    if language == "native":
        return _native_issues(code)
    parsed = _python_issues(code, allowed_extension_modules)
    return parsed if parsed is not None else _native_issues(code)


def _check_issue(code: str, issue: str, message: str) -> tuple[bool, str]:
    return (True, message) if issue in _issues(code) else (False, "")


def check_code_bypass(code: str) -> tuple[bool, str]:
    return _check_issue(code, "code_bypass", _MESSAGES["code_bypass"])


def check_framework_compute(code: str) -> tuple[bool, str]:
    return _check_issue(code, "framework_compute", _MESSAGES["framework_compute"])


def check_stream_injection(code: str) -> tuple[bool, str]:
    return _check_issue(code, "stream_injection", _MESSAGES["stream_injection"])


def check_timing_event_patch(code: str) -> tuple[bool, str]:
    return _check_issue(code, "timing_event_patch", _MESSAGES["timing_event_patch"])


def check_thread_injection(code: str) -> tuple[bool, str]:
    return _check_issue(code, "thread_injection", _MESSAGES["thread_injection"])


def check_lazy_eval(code: str) -> tuple[bool, str]:
    return _check_issue(code, "lazy_eval", _MESSAGES["lazy_eval"])


def check_precision_downgrade(code: str, precision: str = "fp32") -> tuple[bool, str]:
    if normalize_precision(precision) != "fp32":
        return False, ""
    return _check_issue(code, "precision_downgrade", _MESSAGES["precision_downgrade"])


DEFAULT_FORBIDDEN_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",
    "lazy_eval",
    "framework_compute",
    "precision_downgrade",
]
DEFAULT_WARNING_CHECKS = ["stream_injection"]


def validate_kernel_static(
    code: str,
    *,
    precision: str = "fp32",
    forbidden: list[str] | None = None,
    warnings: list[str] | None = None,
    source_map: Mapping[str, str] | None = None,
    allowed_extension_modules: Collection[str] = (),
) -> StaticCheckResult:
    """Validate model code and separately supplied native source files.

    ``code`` remains the model source for backward compatibility.  Callers pass
    only extensions already validated by their precheck as exemptions.
    """
    forbidden_checks = list(DEFAULT_FORBIDDEN_CHECKS if forbidden is None else forbidden)
    warning_checks = list(DEFAULT_WARNING_CHECKS if warnings is None else warnings)
    all_checks, normalized = list(dict.fromkeys(forbidden_checks + warning_checks)), normalize_precision(precision)
    found: set[str] = set()
    snippets = [("python", code)] + [
        ("native" if filename.lower().endswith(_NATIVE_SUFFIXES) else "auto", str(content))
        for filename, content in (source_map or {}).items()
    ]
    for language, snippet in snippets:
        found.update(_issues(snippet, language, allowed_extension_modules if language == "python" else ()))
    if normalized != "fp32":
        found.discard("precision_downgrade")
    errors = [f"{name}: {_MESSAGES[name]}" for name in all_checks if name in found and name in forbidden_checks]
    warnings_out = [
        f"{name}: {_MESSAGES[name]}" for name in all_checks if name in found and name not in forbidden_checks
    ]
    return StaticCheckResult(valid=not errors, errors=errors, warnings=warnings_out, precision=normalized)
