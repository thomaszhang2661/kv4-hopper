"""Run the two diagnostics in one job (one queue wait): IR pipelining check, then loads/layout floor."""
import runpy

runpy.run_path("bench/inspect_ir.py", run_name="__main__")
runpy.run_path("bench/loads_layout.py", run_name="__main__")
