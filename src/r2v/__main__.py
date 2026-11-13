"""`python -m r2v ...` does the same as `r2v ...`.

Needed where Windows Smart App Control blocks the generated r2v.exe launcher.
"""
from r2v.cli import app

app(prog_name="r2v")
