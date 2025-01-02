import importlib.util
spec = importlib.util.spec_from_file_location("Weekly Metrics.PY_PBI0000_FACT_WeeklyMetrics",
                                              'G:/PerfInfo/Performance Management/OR Team/BI Reports/Weekly Metrics/PY_PBI0000_FACT_WeeklyMetrics.py')
module = importlib.util.module_from_spec(spec)
import pandas as pd
spec.loader.exec_module(module)

pivot, original, recent, outliers, recent_trend, correlations, forecasts = module.main()