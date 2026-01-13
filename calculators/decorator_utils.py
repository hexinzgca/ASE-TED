#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# decorator_utils.py

import functools
import inspect
import traceback
from contextlib import ContextDecorator
from typing import Dict, Union
from pprint import pprint
import time  # provides various time-related functions


def debug_helper(enable=True, print_args=True, print_return=True):
    """
    Decorator that combines debug print and entry/exit/exception logging
    Args:
        enable: Controls whether debug output is enabled (True/False)
        print_args: Whether to print function arguments
        print_return: Whether to print function return value
    """
    def decorator(func):
        @functools.wraps(func)  # Preserve original function metadata
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Offer debug print function (outputs based on switch)
            def _debug_print(*msg, **kwargs):
                if enable:
                    # Automatically add function name prefix to distinguish debug info from different functions
                    prefix = f"[{func_name} DEBUG] "
                    print(prefix, *msg, **kwargs)
            
            # Inject _debug_print into the original function's local variables
            # Method: Temporarily add _debug_print by modifying the function's __globals__
            original_globals = func.__globals__.copy()
            func.__globals__['_debug_print'] = _debug_print
            
            try:
                # Print entering function info
                if enable:
                    print(f"\033[94m[DEBUG] Entering function: {func_name}()\033[0m")  # Blue
                
                # Print arguments
                if enable and print_args:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    args_info = ', '.join([f"{k}={v}" for k, v in bound_args.arguments.items()])
                    print(f"[DEBUG] Function arguments: {args_info}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Print leaving function and return value
                if enable:
                    print(f"\033[92m[DEBUG] Leaving function: {func_name}()\033[0m")  # Green
                    if print_return:
                        print(f"[DEBUG] Return value: {result}")
                return result
            
            except Exception as e:
                # Print exception info (red)
                if enable:
                    print(f"\033[91m[DEBUG] Function {func_name}() threw exception:\033[0m")
                    print(f"[DEBUG] Exception type: {type(e).__name__}")
                    print(f"[DEBUG] Exception info: {str(e)}")
                    print(f"[DEBUG] Stack trace:\n{traceback.format_exc()}")  # Full stack
                raise  # Re-raise exception, don't interrupt original exception handling flow
            
            finally:
                # Restore original function's global variables (avoid side effects)
                func.__globals__.update(original_globals)
        
        return wrapper
    return decorator
    

class FinalMeta(type):
    """
    Metaclass: Prevents subclasses from overriding methods marked with @FinalMeta.final_method
    Example: Using decorator and metaclass to protect parent class methods
    ```
    class Parent(metaclass=FinalMeta):
        @FinalMeta.final_method  # Mark as forbidden to override
        def critical_method(self):
            print("This is the parent class's core method, forbidden for subclasses to modify")
    ```
    """

    @classmethod
    def final_method(cls, func):
        """Decorator: Marks method as 'forbidden to be overridden by subclasses'"""
        func.__is_final__ = True  # Add marker
        return func

    def __new__(cls, name, bases, namespace):
        # Iterate through methods marked with @final_method in parent classes (bases)
        for base in bases:
            # Get all method names in parent class marked as 'forbidden to override'
            final_methods = {
                attr: method
                for attr, method in base.__dict__.items()
                if hasattr(method, '__is_final__') and method.__is_final__
            }
            # Check if subclass overrides these methods
            for attr, method in final_methods.items():
                if attr in namespace:
                    # Raise error to prevent subclass definition
                    raise TypeError(
                        f"Subclass '{name}' cannot override protected method '{attr}' (marked with @final_method)"
                    )
        # Create subclass normally
        return super().__new__(cls, name, bases, namespace)

# Define a ContextDecorator class for timing code sections
class Timing(ContextDecorator):
    # Class variable to store timing information
    timers: Dict[str, Union[float, float, int]] = dict()

    def __init__(self, name: Union[str, None] = None):
        """
        Initialize the Timing object with an optional name.
        
        :param name: The name of the code section to be timed.
        """
        self._start_CPU_time = None  # CPU time at the start of the timing
        self._start_wall_time = None  # Wall time at the start of the timing
        self.name = name  # Name of the code section
        if self.name:
            # Initialize the timer for the given section if it doesn't exist
            self.timers.setdefault(self.name, [0.0, 0.0, 0])

    def __enter__(self):
        """
        Start timing when entering the context.
        
        :return: The Timing object itself.
        """
        self._start_CPU_time = time.process_time()  # Start CPU time measurement
        self._start_wall_time = time.perf_counter()  # Start wall time measurement
        return self

    def __exit__(self, *exc_info):
        """
        Stop timing when exiting the context and accumulate the elapsed time.
        
        :param exc_info: Exception information (if any) that occurred within the context.
        :return: A tuple containing the CPU and wall elapsed times.
        """
        # Calculate elapsed time
        CPU_elapsed_time = time.process_time() - self._start_CPU_time
        wall_elapsed_time = time.perf_counter() - self._start_wall_time
        # Accumulate elapsed time
        if self.name:
            self.timers[self.name][0] += CPU_elapsed_time  # Accumulate CPU time
            self.timers[self.name][1] += wall_elapsed_time  # Accumulate wall time
            self.timers[self.name][2] += 1  # Increment the call count

        if exc_info[0]:
            pprint(exc_info[1])
            return False 
        return True

    @staticmethod
    def report(userprint=None):
        """
        Generate and print a report of the accumulated timing information.
        
        :param userprint: A function to use for printing the report. If None, print to stdout.
        """
        # Sort the timers by wall time in descending order
        sorted_timers = sorted(Timing.timers.items(), key=lambda kv: kv[1][1], reverse=True)
        log = ""
        log += "===============================================================================================\n"
        log += " Code section                           | total WallTime/ s   | total CPUTime/ s    | No. calls      \n"
        log += "-----------------------------------------------------------------------------------------------\n"
        for section, times in sorted_timers:
            cputime, walltime, nrcalls = times
            if nrcalls == 0: continue  # Skip sections that haven't been called
            log += " {0:35s}    | {1:15.2f}     | {2:15.2f}     | {3:8d}       \n".format(
                section, walltime, cputime, nrcalls)
        log += "===============================================================================================\n\n"
        if userprint is None:
            print(log)  # Print the report to stdout
        else:
            userprint(log)  # Use the provided function to print the report