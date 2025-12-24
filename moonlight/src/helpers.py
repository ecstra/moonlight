import logging, time
from rich.console import Console
from typing import List
from statistics import mean

logger = logging.getLogger(__name__)
console = Console()

def log_warning(message: str):
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")

def log_error(message: str):
    console.print(f"[bold red]✕[/bold red] {message}")

def log_info(message: str):
    console.print(f"[bold blue]info:[/bold blue] {message}")

def discover_agents(parameters: dict = {}):
    """
    Generate a prompt for the parameters of a function.
    """
    
    if not parameters:
        return "", {}
        
    prompt = "\n\nYou have access to the following parameters:\n```\n"
    for key, value in parameters.items():
        value_type = type(value).__name__
        prompt += f"- {key}: type {value_type}\n"
    prompt += "```\nYou can use these parameters if needed, otherwise generate your own."
    return prompt

class TimeTracker:
    def __init__(self):
        self.times: List[float] = []  # stored in seconds
        self._start_time: int = 0    # nanoseconds
        self._is_measuring: bool = False
        self.total_items: int = 0
        self.completed_items: int = 0
        self.total_elapsed_time = 0.0

    def begin_measure(self) -> None:
        """Begin a measurement run"""
        if not self._is_measuring:
            self._start_time = time.perf_counter_ns()
            self._is_measuring = True

    def end_measure(self) -> float:
        """End the measurement run and record elapsed time in seconds"""
        if self._is_measuring:
            elapsed_ns = time.perf_counter_ns() - self._start_time
            elapsed_time = elapsed_ns / 1e9
            self.times.append(elapsed_time)
            self._is_measuring = False
            self.completed_items += 1
            self.total_elapsed_time += elapsed_time
            return elapsed_time
        return 0.0

    def get_average_time(self) -> float:
        """Return the average recorded time in seconds"""
        return mean(self.times) if self.times else 0.0

    def set_total_items(self, total: int) -> None:
        """Set the total number of items to process"""
        self.total_items = total

    def get_estimated_remaining_time(self) -> float:
        """Estimate remaining time based on the average time and remaining items"""
        if not self.times or self.total_items <= self.completed_items:
            return 0.0
        avg_time = self.get_average_time()
        remaining_items = self.total_items - self.completed_items
        return avg_time * remaining_items

    def reset(self) -> None:
        """Reset all measurement data"""
        self.times.clear()
        self._start_time = 0
        self._is_measuring = False
        self.completed_items = 0

    def get_stats(self) -> dict:
        """Return current timing statistics"""
        return {
            'average_time': self.get_average_time(),
            'total_samples': len(self.times),
            'estimated_remaining_time': self.get_estimated_remaining_time(),
            'completed_items': self.completed_items,
            'total_items': self.total_items,
            'total_elapsed_time': self.total_elapsed_time,
        }