"""EDA entrypoint — redirige al hardening/análisis Fase 2."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "preprocessing_eda" / "harden_fase2_cti.py"), run_name="__main__")
