import importlib.util
spec = importlib.util.spec_from_file_location("Weekly Metrics.PY_PBI0000_FACT_WeeklyMetrics",
                                              'G:/PerfInfo/Performance Management/OR Team/BI Reports/Weekly Metrics/PY_PBI0000_FACT_WeeklyMetrics.py')
module = importlib.util.module_from_spec(spec)
import pandas as pd
spec.loader.exec_module(module)

full_data, historical, last_7_days, outliers, recent_trend, correlations, forecasts, metrics = module.main()