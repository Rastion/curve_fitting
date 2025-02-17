import math
import random
from qubots.base_problem import BaseProblem

class CurveFittingProblem(BaseProblem):
    """
    Curve Fitting Problem:
    Given a set of input/output observations, find the parameters a, b, c, d 
    for the mapping function:
    
         f(x) = a * sin(b - x) + c * x^2 + d
         
    that minimize the sum of squared errors between f(x) and the observed outputs.
    """

    def __init__(self, inputs=None, outputs=None, instance_file=None):
        """
        Initialize the curve fitting problem.
        
        Provide either:
          - `instance_file`: a path to a file containing instance data, or
          - `inputs` and `outputs`: lists of observed values.
        
        The instance file format is expected to be:
          [number_of_observations]
          [x1] [y1]
          [x2] [y2]
          ...
          [xn] [yn]
        """
        if instance_file is not None:
            self._load_instance_from_file(instance_file)
        else:
            if inputs is None or outputs is None:
                raise ValueError("Either 'instance_file' or both 'inputs' and 'outputs' must be provided.")
            if len(inputs) != len(outputs):
                raise ValueError("The number of inputs must equal the number of outputs.")
            self.inputs = inputs
            self.outputs = outputs
        self.nb_observations = len(self.inputs)

    def _load_instance_from_file(self, filename):
        with open(filename, "r") as f:
            tokens = f.read().split()
        # Convert tokens to floats
        values = [float(tok) for tok in tokens]
        nb = int(values[0])
        expected_len = 1 + nb * 2
        if len(values) != expected_len:
            raise ValueError(f"Instance file format error: expected {expected_len} numbers, got {len(values)}.")
        self.inputs = []
        self.outputs = []
        index = 1
        for _ in range(nb):
            self.inputs.append(values[index])
            self.outputs.append(values[index + 1])
            index += 2

    def evaluate_solution(self, solution) -> float:
        """
        Given a candidate solution (a, b, c, d), compute the sum of squared errors.
        """
        if not isinstance(solution, (list, tuple)) or len(solution) != 4:
            raise ValueError("Solution must be a list or tuple of 4 parameters: [a, b, c, d].")
        a, b, c, d = solution
        total_error = 0.0
        for x, y in zip(self.inputs, self.outputs):
            # Mapping function: f(x) = a*sin(b - x) + c*x^2 + d
            prediction = a * math.sin(b - x) + c * (x ** 2) + d
            error = prediction - y
            total_error += error ** 2
        return total_error

    def random_solution(self):
        """
        Generate a random candidate solution with parameters in the range [-100, 100].
        """
        return [random.uniform(-100, 100) for _ in range(4)]
