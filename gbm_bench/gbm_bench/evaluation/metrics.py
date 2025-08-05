import numpy as np
from typing import Dict, List, Union, Tuple


def coverage(recurrence: np.ndarray, plan: np.ndarray) -> float:
     """
     Computes coverage in % of the recurrence by the plan.

     Parameters:
         recurrence (np.ndarray):
         plan (np.ndarray):
     
     Returns:
         float: The coverage [0,1]
     """
     if np.sum(recurrence) <=  0.00001:
         return 1

     intersection = np.logical_and(recurrence, plan)
     coverage = np.sum(intersection) / np.sum(recurrence)
     return coverage
