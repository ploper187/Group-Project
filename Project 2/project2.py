__author__ = "Cameron Scott, Adam Kuniholm, and Sam Fite"
__right__ = "right 2019, Cameron Scott, Adam Kuniholm, and Sam Fite"
__credits__ = ["Cameron Scott", "Adam Kuniholm", "Sam Fite"]
__license__ = "MIT"

LIMIT_TIME = 999

from statistics import mean
import sys
import random
import math
from copy import copy
from enum import Enum
import time
import collections
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
        s = " ".join([self.timestamp_str() \
                    , "Simulator" \
                    , self.action \
                    , "(" + self.simulation.memoryType \
                    , "--" \
                    , self.simulation.memoryAlgorithm + ")"])
        return s


class StartSimulation(SimulationEvent): action = "started"
class EndSimulation(SimulationEvent):   action = "ended"

class ProcessEvent(Event):
    process = None
    simulation = None
    
    def __init__(self, simulation, process):
        self.process = (process)
        self.simulation = (simulation)

class ProcessPlaced(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str(), "Placed process", self.process.name + "\n" + str(self.simulation)])


class ProcessArrival(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str(), "Process", self.process.name, "arrived", "(requires", self.process.num_frames, "frames)"])

class ProcessTermination(ProcessEvent):
    def __str__(self): return " ".join(
        [self.timestamp_str(), "Process", self.process.name, "removed:\n" + str(self.simulation)])


class Process:
    Burst = collections.namedtuple('Burst', 'start_time duration')
    name = None
    events = None
    num_frames = None
    frames = None
    bursts = None # [(arrival_time, run_time)]
    num_bursts_left = None
    def __init__(self, name, frames_needed, bursts):
        self.name = name
        self.events = []
        self.num_frames = frames_needed
        self.bursts = bursts
        self.num_bursts_left = len(bursts)
        self.frames = []
    def __str__(self): 
        return "Process {} -- ({}){} -- ({}){}".format(\
                self.name, self.num_frames, [str(f) for f in self.frames], \
                len(self.bursts), [(s.start_time, s.duration) for s in self.bursts])

    def add_frame(self, frame):   self.frames.append(frame)
    def set_frames(self, frames): self.frames = frames
    def clear_frames(self):       self.frames = []

class ProcessFactory: pass
class InputFileProcessFactory(ProcessFactory):
    # Uncomment the next line if using Python 2.x...
    # from __future__ import division
    @staticmethod
    def generate(input_filename): 
        processes = []
        with open(input_filename) as file:
            for line in file:
                if (line[0] == "#"):
                    continue
                l = line.split()
                name = l[0]
                num_frames = int(l[1])
                bursts = []
                for times in l[2:]:
                    tl = times.split("/")
                    bursts.append(Process.Burst(start_time=int(tl[0]),duration=int(tl[1])))
                processes.append(Process(name, num_frames, bursts))
                # print("{} - {} w/ {}, {}".format( input_filename, name, frames_reqd, time_slots))
        return processes

class Frame:
    process = None
    address = None
    simulator = None
    def __init__(self, simulator, address, process = None):
        self.process = process
        self.simulator = simulator
        self.address = address
    def process_name(self):
        return self.process.name if self.process is not None else "."
    def __str__(self):
        return "Frame -- {} -- {} -- {}".format(str(self.process), self.address, self.simulator.name)
    def is_free(self): return self.process == None
    def free(self):
        if (self.is_free()): 
            return False
        self.process = None
        self.address = None
        return True
    def allocate(self, process, at_address, force = False):
        if ((not force) and self.is_free()):
            return False
        self.process = process
        self.location = int(at_address)
        return True
    def move(self, address):
        self.address = address
    def reallocate(self, new_process, at_new_address):
        if (not self.is_free()):
            self.free()
        return self.allocate(new_process, at_new_address)


class MemorySimulator:
    memory_type = "No memory type"
    memory_algorithm = "No memory algorithm"
    num_frames_per_line = None #0
    num_frames_tot = None #0
    num_frames_allocated = None #0
    frames = []
    # Ready processes
    ready = None #[]
    # Completed processes
    completed = None #[]
    # Events
    events = None #[]
    # Logs
    log = None #[]
    # Current time (in ms)
    current_time = None #0
    # Queue
    queue = None #[]
    # Number of processes that the schedule started with
    num_processes = None #0

    def execute(self): pass
    def allocate(self, process): pass
    def free_block(self, range): pass
    

    def __str__(self):
        s = "="*self.num_frames_per_line
        for i in range(len(self.frames)):
            s += str(self.frames[i])
            if (i%self.num_frames_per_line == 0):
                s += "\n"
        s = "="*self.num_frames_per_line
        return s


    def __init__(self, processes, num_frames, num_frames_per_line = 32):
        self.num_frames_per_line = num_frames_per_line
        self.num_frames_tot = num_frames
        self.num_frames_allocated = 0
        self.queue = processes
        self.num_processes = len(self.queue)
        self.ready = []
        self.log = []
        self.events = []
        self.current_time = 0
        self.completed = []

    def is_completed(self): 
        return len(self.completed) == self.num_processes
    def logs(self):
        ret = sorted(self.log, key = lambda x: (x[1], x[2]))
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
        if (isinstance(event, ProcessEvent)):
            event.process.events.append(event)
        self.events.append(event)
        t = type(event).__name__
        self.log.append((str(event), timestamp, t))
    def begin(self): self.log_event(StartSimulation(self))
    def end(self):   self.log_event(EndSimulation(self))
    def process_arrived(self, process):
        # TODO: Configure process arrival
        self.log_event(ProcessArrival(self, process))
    def process_terminated(self, process):
        # TODO: Configure process termination
        process.completion_ts = self.current_time
        self.log_event(ProcessTermination(self, process))

class ContiguousMemorySimulator(MemorySimulator): 
    memory_type = "Contiguous"
    def allocate(self, process): pass
    def defragment(self): pass


class NonContiguousMemorySimulator(MemorySimulator): 
    memory_type = "Noncontiguous"
    def allocate(self, process): pass

class FFMemorySimulator(ContiguousMemorySimulator): pass
class NFMemorySimulator(ContiguousMemorySimulator): pass
class BFMemorySimulator(ContiguousMemorySimulator): pass 

'''#############################################################################
#                                     MAIN                                     #
# '''
if __name__ == '__main__':
    is_debug = len(sys.argv) > 0 and sys.argv[-1] == "debug"
    argv = sys.argv
    if (len(sys.argv) < 5):
        print("Not enough arguments. {}/5 provided!".format(len(sys.argv)))
        sys.exit() 
    num_frames_per_line = int(argv[1])
    num_frames = int(argv[2])
    input_filename = argv[3]
    t_memmove = argv[4]
    processes = InputFileProcessFactory.generate(input_filename)  
    for p in processes:
        print(p)   
    




    # Generate simulations
    
    # Queue processes
    
    # Create and execute simulations

    # Output simulation logs



    # Done
