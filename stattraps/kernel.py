"""
A persistent Python process for the agent's code (variables survive between tool calls, like a notebook).

Protocol: the parent writes one JSON line {"code": "..."}; the kernel executes it in a shared namespace
with stdout/stderr captured and answers with one JSON line {"output": "...", "error": bool}.
Run as:  python -m stattraps.kernel   (cwd = the run's workspace)
"""
import ast
import contextlib
import io
import json
import sys
import traceback


def run_cell(code: str, ns: dict) -> None:
    """Execute code; if the last statement is an expression, print its value (like a notebook cell)."""
    tree = ast.parse(code)
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr):
        tree.body = tree.body[:-1]
        exec(compile(tree, "<cell>", "exec"), ns)
        value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)
        if value is not None:
            print(repr(value))
    else:
        exec(compile(tree, "<cell>", "exec"), ns)


if __name__ == "__main__":
    out = sys.__stdout__
    namespace = {"__name__": "__main__"}
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass
    for line in sys.stdin:
        try:
            code = json.loads(line)["code"]
        except Exception:
            continue
        buf, err = io.StringIO(), False
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                run_cell(code, namespace)
            except BaseException:
                err = True
                traceback.print_exc()
        out.write(json.dumps({"output": buf.getvalue(), "error": err}) + "\n")
        out.flush()
