@echo off
call C:\Users\qwqw1\anaconda3\Scripts\activate.bat fintech
python -m pytest tests/test_gui_controller.py::TestSweepRunnerCallbacks::test_sweep_runner_backward_compatible -v
