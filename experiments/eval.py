import numpy as np

from utils.globals import ReportState

def length_score(target, sigma) -> float:
    length = len(ReportState.instance().content)

    return np.exp(-0.5 * ((length - target) / sigma)**2)