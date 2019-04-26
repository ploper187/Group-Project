__author__ = "Cameron Scott, Adam Kuniholm, and Sam Fite"
__right__ = "right 2019, Cameron Scott, Adam Kuniholm, and Sam Fite"
__credits__ = ["Cameron Scott", "Adam Kuniholm", "Sam Fite"]
__license__ = "MIT"
'''
CS 4210 Operating Systems
Group Project 1 - CPU Scheduling Simulation

• This project is due by 11:59:59 PM on Friday, March 22, 2019.
• As per usual, your code must successfully compile/run on Submitty,
  which uses Ubuntu v18.04.1 LTS.
• For Python, you must use python3, which is Python 3.6.7.
  Be sure to name your main Python file project1.py

== Project specifications ==
In this first project, you will implement a rudimentary simulation of an operating
system. The initial focus will be on processes, assumed to be resident in memory,
waiting to use the CPU. Memory and the I/O subsystem will not be covered in depth
in this project.

'''

'''#############################################################################
#                                   IMPORTS                                    #
# '''
import collections
import time
from enum import Enum
from copy import copy
import math
import random
import sys
from statistics import mean
LIMIT_TIME = 999

'''#############################################################################
#                              CLASS DECLARATIONS                              #
# '''

class Event:
    timestamp = -1
    def timestamp_str(self): return "time " + str(self.timestamp) + "ms:"
    def __str__(self):
        return "An Event"

'''
time <t>ms: <event-details> [Q <queue-contents>]

• Start of simulation
• Process arrival
• Process starts using the CPU
• Process finishes using the CPU (i.e., completes a CPU burst)
• (v1.3) Process has its τ value recalculated (i.e., after a CPU burst completion)
• Process preemption
• Process starts performing I/O
• Process finishes performing I/O
• Process terminates by finishing its last CPU burst
• End of simulation
'''

class SimulationEvent(Event):
    simulation = None
    action = "<action not set>"
    def __init__(self, simulation): self.simulation = (simulation)

    def __str__(self):
        # time 0ms: Simulator started (Contiguous -- First-Fit)
        s = " ".join([self.timestamp_str(), "Simulator", self.action, "(" +
                      self.simulation.memoryType, "--", self.simulation.memoryAlgorithm + ")"])
        return s

class StartSimulation(SimulationEvent): action = "started"
class EndSimulation(SimulationEvent):   action = "ended"

class BurstEvent(Event):
    burst = None
    simulation = None
    def __init__(self, simulation, burst):
        self.burst = burst
        self.simulation = simulation

class BurstArrival(BurstEvent):
    def __str__(self): return " ".join([self.timestamp_str(), "Process", self.burst.process_name, "arrived",
                                        "(requires", self.burst.num_frames, "frames)"])

class BurstTermination(BurstEvent):
    def __str__(self): return " ".join(
        [self.timestamp_str(), "Process", self.burst.process_name, "removed:\n" + str(self.simulation)])

class Burst:
    start_time = None
    duration = None
    num_frames = None
    process_name = None
    allocation_start_idx = None
    allocation_end_idx = None
    def __init__(self, start_time, duration, num_frames, process_name):
        self.start_time = start_time
        self.duration = duration
        self.num_frames = num_frames
        self.allocation_start_idx = None
        self.allocation_end_idx = None

class BurstFactory: pass
class InputFileBurstFactory(BurstFactory):
    @staticmethod
    def generate(input_filename):
        bursts = []
        with open(input_filename) as file:
            for line in file:
                if (line[0] == "#"):
                    continue
                split_line = line.split()
                name = split_line[0]
                num_frames = int(split_line[1])
                for times in split_line[2:]:
                    tl = times.split("/")
                    start_time = int(tl[0])
                    duration = int(tl[1])
                    b = Burst(start_time, duration, num_frames, name)
                    bursts.append(b)
        return bursts

class MemorySimulator:
    memory_type = "No memory type"
    memory_algorithm = "No memory algorithm"
    num_frames_per_line = None  # 0
    num_frames_tot = None  # 0
    num_frames_allocated = None  # 0
    frames = []
    # Ready processes
    ready = None  # []
    # Completed processes
    completed = None  # []
    # Events
    events = None  # []
    # Logs
    log = None  # []
    # Current time (in ms)
    current_time = None  # 0
    # Queue
    queue = None  # []
    # Number of processes that the schedule started with
    num_bursts = None  # 0

    def execute(self): pass
    def allocate(self, burst): pass
    def free(self, burst): pass

    def __str__(self):
        s = "="*self.num_frames_per_line
        for i in range(len(self.frames)):
            if (i % self.num_frames_per_line == 0):
                s += "\n"
            s += self.frames[i]

        s += "\n" + "="*self.num_frames_per_line
        return s

    def __init__(self, bursts, num_frames, num_frames_per_line=32):
        self.num_frames_per_line = num_frames_per_line
        self.num_frames_tot = num_frames
        self.num_frames_allocated = 0
        self.queue = bursts
        self.num_bursts = len(self.queue)
        self.log = []
        self.events = []
        self.current_time = 0
        self.completed = []
        self.frames = ['.' for i in range(num_frames)]

    def is_completed(self):
        return len(self.completed) == self.num_bursts

    def logs(self):
        ret = sorted(self.log, key=lambda x: (x[1], x[2]))
        string = ""
        for event in ret:
            string += event[0] + ("\n" if event is not ret[-1] else "")
        return string
    '''
        Log an event, such as a preemption, process completion, or context switch
        e.g. self.log_event(ProcessCompleted(p1))
    '''

    def log_event(self, event):
        timestamp = self.current_time
        event.timestamp = self.current_time
        self.events.append(event)
        t = type(event).__name__
        self.log.append((str(event), timestamp, t))

    def begin(self): self.log_event(StartSimulation(self))
    def end(self):   self.log_event(EndSimulation(self))

    def burst_started(self, burst):
        # TODO: Configure process arrival
        self.log_event(BurstArrival(self, burst))

    def burst_ended(self, burst):
        # TODO: Configure process termination
        burst.completion_ts = self.current_time
        self.log_event(BurstTermination(self, burst))

class ContiguousMemorySimulator(MemorySimulator): 
    memory_type = "Contiguous"

class NonContiguousMemorySimulator(MemorySimulator):
    memory_type = "Noncontiguous"
    memory_algorithm = "Best-Fit"

    def execute(self):
        """
        TODO:
        Run through the bursts here, calling allocate and free
        when appropriate
        """

    def allocate(self, burst):
        """
        TODO:
        Allocate memory for a burst here, according to the 
        algorithm, and defragment whenever necessary
        """

    def free(self, burst):
        """
        TODO:
        Deallocate memory of a burst here, according to the
        algorithm, and defragment whenever necessary
        """


class FFMemorySimulator(ContiguousMemorySimulator):
    memory_algorithm = "First-Fit"

    def execute(self):
        """
        TODO:
        Run through the bursts here, calling allocate and free
        when appropriate
        """

    def allocate(self, burst):
        """
        TODO:
        Allocate memory for a burst here, according to the 
        algorithm, and defragment whenever necessary
        """

    def free(self, burst):
        """
        TODO:
        Deallocate memory of a burst here, according to the
        algorithm, and defragment whenever necessary
        """


class NFMemorySimulator(ContiguousMemorySimulator):
    memory_algorithm = "Next-Fit"

    def execute(self):
        """
        TODO:
        Run through the bursts here, calling allocate and free
        when appropriate
        """

    def allocate(self, burst):
        """
        TODO:
        Allocate memory for a burst here, according to the 
        algorithm, and defragment whenever necessary
        """

    def free(self, burst):
        """
        TODO:
        Deallocate memory of a burst here, according to the
        algorithm, and defragment whenever necessary
        """


class BFMemorySimulator(ContiguousMemorySimulator):
    memory_algorithm = "Best-Fit"

    def execute(self):
        """
        TODO:
        Run through the bursts here, calling allocate and free
        when appropriate
        """

    def allocate(self, burst):
        """
        TODO:
        Allocate memory for a burst here, according to the 
        algorithm, and defragment whenever necessary
        """

    def free(self, burst):
        """
        TODO:
        Deallocate memory of a burst here, according to the
        algorithm, and defragment whenever necessary
        """


'''#############################################################################
##                                    MAIN                                    ##
#############################################################################'''
if __name__ == '__main__':
    is_debug = len(sys.argv) > 0 and sys.argv[-1] == "debug"
    argv = sys.argv
    if (len(sys.argv) < 5):
        raise Exception('Not enough arguments. 5 needed!')
    num_frames_per_line = int(argv[1])
    num_frames = int(argv[2])
    input_filename = argv[3]
    t_memmove = int(argv[4])
    # Generate processes
    bursts = InputFileBurstFactory.generate(input_filename)
    # Create simulations
    ff = FFMemorySimulator(bursts, num_frames, num_frames_per_line)
    nf = NFMemorySimulator(bursts, num_frames, num_frames_per_line)
    bf = BFMemorySimulator(bursts, num_frames, num_frames_per_line)
    nc = NonContiguousMemorySimulator(bursts, num_frames, num_frames_per_line)
    # Execute simulations
    ff.execute()
    nf.execute()
    bf.execute()
    nc.execute()
    # Output simulation logs
    print(ff.logs())
    print(nf.logs())
    print(bf.logs())
    print(nc.logs())
    # Done
