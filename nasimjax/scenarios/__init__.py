  # Copyright (c) 2018 Jonathon Schwartz (NASim).                                         
  # Licensed under the MIT License. See LICENSE-NASIM in the project root.                  
  #
  # This file is included unmodified from NASim                                             
  # (https://github.com/Jjschwartz/NetworkAttackSimulator) at commit 4f26de3,              
  # file nasim/scenarios/__init__.py.    

from nasimjax.scenarios.loader import ScenarioLoader
import nasimjax.scenarios.benchmark as benchmark


def make_benchmark_scenario(scenario_name):
    """Generate or Load a benchmark Scenario.

    Parameters
    ----------
    scenario_name : str
        the name of the benchmark environment

    Returns
    -------
    Scenario
        a new scenario instance

    Raises
    ------
    NotImplementederror
        if scenario_name does no match any implemented benchmark scenarios.
    """
    if scenario_name in benchmark.AVAIL_STATIC_BENCHMARKS:
        scenario_def = benchmark.AVAIL_STATIC_BENCHMARKS[scenario_name]
        return load_scenario(scenario_def["file"], name=scenario_name)
    else:
        raise NotImplementedError(
            f"Benchmark scenario '{scenario_name}' not available."
            f"Available scenarios are: {benchmark.AVAIL_BENCHMARKS}"
        )


def load_scenario(path, name=None):
    """Load NASim Environment from a .yaml scenario file.

    Parameters
    ----------
    path : str
        path to the .yaml scenario file
    name : str, optional
        the scenarios name, if None name will be generated from path
        (default=None)

    Returns
    -------
    Scenario
        a new scenario object
    """
    loader = ScenarioLoader()
    return loader.load(path, name=name)


def get_scenario_max(scenario_name):
    if scenario_name in benchmark.AVAIL_STATIC_BENCHMARKS:
        return benchmark.AVAIL_STATIC_BENCHMARKS[scenario_name]["max_score"]
    return None
