"""
Full factor-research report: Fama-MacBeth premia → IC analysis → turnover/capacity.

    python run_report.py

Runs the three analyses in order on the cached, survivorship-free factor panel
(56,915 stock-months, 2016-2026). This is the statistical-alpha layer over the
factor sleeve: the academic-quant evaluation a research desk runs before sizing a
signal — factor premia with Newey-West t-stats, predictive-power decay, and the
turnover/cost/capacity profile.
"""

import fama_macbeth
import ic_analysis
import capacity

if __name__ == "__main__":
    fama_macbeth.report()
    ic_analysis.report()
    capacity.main()
