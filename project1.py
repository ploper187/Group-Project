__author__ = "Cameron Scott, Adam Kuniholm, and Sam Fite"
__copyright__ = "Copyright 2019, Cameron Scott, Adam Kuniholm, and Sam Fite"
__credits__ = ["Cameron Scott", "Adam Kuniholm", "Sam Fite"]
__license__ = "MIT"


from enum import Enum
from copy import deepcopy
import math
import random
import sys
'''
CS 4210 Operating Systems
Group Project 1 - CPU Scheduling Simulation

• This project is due by 11:59:59 PM on Monday, March 18, 2019.
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
#############################################################################'''
'''#############################################################################
#                              CLASS DECLARATIONS                              #
#############################################################################'''
class Event:
    timestamp = -1
    ready_queue = []
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
class StartSimulation(Event):
    simulation = None
    def __str__(self): return "time 0ms: Simulator started for FCFS [Q <empty>]"
class EndSimulation(Event):
    simulation = None
    def __str__(self): return "time 475484ms: Simulator ended for SRT [Q <empty>]"

class ProcessEvent(Event):
    process = None
    def __init__(self, process):
        self.process = process
class ProcessArrival(ProcessEvent):
    def __str__(self): return "time 40ms: Process B (tau 1000ms) arrived; added to ready queue [Q B]"
class CPUBurstBegun(ProcessEvent):
    def __str__(self): return "time 42ms: Process B started using the CPU for 505ms burst [Q <empty>]"
class CPUBurstEnded(ProcessEvent):
    def __str__(self): return "time 547ms: Process B completed a CPU burst; 88 bursts to go [Q A F H I K]"
class ProcessTauRecalculated(ProcessEvent):
    def __str__(self): return "time 547ms: Recalculated tau = 753ms for process B [Q A F H I K]"
class Preemption(ProcessEvent):
    def __str__(self): return "time 104239ms: Process G (tau 434ms) completed I/O and will preempt D [Q K E F H C B]"
class IOBurstStarts(ProcessEvent):
    def __str__(self): return "time 105789ms: Process G switching out of CPU; will block on I/O until time 105805ms [Q J A D K E F H C B]"
class IOBurstEnds(ProcessEvent):
    def __str__(self): return "time 104607ms: Process A (tau 1104ms) completed I/O; added to ready queue [Q A D K E F H C B]"
class ProcessCompleted(ProcessEvent):
    def __str__(self): return "time 475482ms: Process B terminated [Q <empty>]"


class Process:
    class State(Enum):
        RUNNING = 0
        READY = 1
        BLOCKED = 2
        COMPLETED = -1
        
    name = None
    creation_ts = -1
    last_active_ts = -1
    last_blocked_ts = -1
    bursts_remaining = -1
    cpu_burst_time = -1
    io_burst_time = -1
    
    state = State.READY

    def __init__(self, timestamp, 
                 num_bursts, cpu_burst_time, io_burst_time):
        self.creation_ts = timestamp
        self.bursts_remaining = num_bursts

        self.cpu_burst_time = cpu_burst_time

        self.io_burst_time = io_burst_time

    def time_remaining(self):
        return self.bursts_remaining*(self.cpu_burst_time + self.io_burst_time)

    def is_completed(self):
        return self.bursts_remaining > 0




class Scheduler:
    # Active process (using the CPU now)
    active = None
    # Blocked processes
    blocked = []
    # Ready processes
    ready = []
    # Completed processes
    completed = []
    # Events
    events = []
    # Current time (in ms)
    current_time = 0
    # Total amount of time processes bursted (ms)
    cpu_burst_time = 0
    # Total amount of time processes waited (ms)
    tot_wait_time = 0
    # Total turnaround time (ms)
    tot_turnaround_time = 0


    def execute(self): pass
    def __init__(self, processes): self.ready = processes


    def __str__(self):
        n = len(self.completed)
        s =  '-- average CPU burst time: {0} ms\n'.format(self.cpu_burst_time/n)
        s += '-- average wait time: {0} ms\n'.format(self.tot_wait_time/n) 
        s += '-- average turnaround time: {0} ms\n'.format(self.tot_turnaround_time/n)
        s += '-- total number of context switches: {0}\n'.format(len(filter(lambda cs: cs is Preemption, self.events)))
        s += '-- total number of preemptions: {0}\n'.format(len(filter(lambda cs: cs is Preemption, self.events)))
        return s

    def is_completed(self):
        return self.ready.isempty()
    def logs(self):
        s = ''
        for event in self.events:
            s += str(event) 
        return s
    '''
        Log an event, such as a preemption, process completion, or context switch

        e.g. self.log_event(ProcessCompleted(p1))
    '''
    def log_event(self, event):
        event.queue = deepcopy(self.ready)
        self.events.append(event)
    def begin(self):
        self.log_event(StartSimulation(self))
    def end(self):
        self.log_event(EndSimulation(self))
    def process_arrived(self, process):
        self.log_event(ProcessArrival(process))
    def process_burst(self, process):
        self.log_event(CPUBurstBegun(process))
    def process_ends_burst(self, process):
        self.log_event(CPUBurstEnded(process))
    def recaculated_tau(self, process):
        self.log_event(ProcessTauRecalculated(process))
    def process_io_burst(self, process):
        self.log_event(IOBurstStarts(self))
    def process_ends_io_burst(self, process):
        self.log_event(IOBurstEnds(process))
    
''' Shorted Job First Process Scheduler '''
class SJFScheduler(Scheduler):
    name = "SJF"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        pass


'''Shortest Time Remaining Process Scheduler'''
class SRTScheduler(Scheduler):
    name = "SRT"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        pass


''' First Come First Server Process Scheduler '''
class FCFSScheduler(Scheduler):
    name = "FCFS"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        pass

''' Round Robin Process Scheduler '''
class RRScheduler(Scheduler):
    name = "RR"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        pass

class ProcessFactory:
    def generate(self):
        return []

class RandomProcessFactory(ProcessFactory):
    ics = None
    def __init__(self, argv):
        self.ics = RandomProcessFactory.InitialConditions(argv)
    
    def generate(self):
        random.seed(self.ics.seed)
        processes = []
        for _ in range(self.ics.num_processes):
            arrival_time = -1
            while (arrival_time < 0 or arrival_time > self.ics.arrival_time_cieling):
                arrival_time = self.ics.map_to_exp(random.random())
            burst_time = self.ics.map_to_exp(random.random())
            io_burst_time = self.ics.map_to_exp(random.random())

            num_bursts = math.trunc(random.random()*100)+1
            process = Process(arrival_time, num_bursts, burst_time, io_burst_time)
            processes.append(process)
        return processes

    class InitialConditions:
        seed = -1
        lambda_value = -1
        arrival_time_cieling = -1
        num_processes = -1
        context_switch_duration = -1
        alpha = -1
        time_slice = -1
        rr_should_push_front = False
        def map_to_exp(self, uniform_value):
            return (-1)*math.log2(uniform_value)/self.lambda_value

        def __init__(self, argv):
            self.seed = argv[1]
            self.lambda_value = argv[2]
            self.arrival_time_cieling = argv[3]
            self.num_processes = argv[4]
            self.context_switch_duration = argv[5]
            self.alpha = argv[6]
            self.time_slice = argv[7]
            self.rr_should_push_front = True if argv[8] == 'BEGINNING' else False


'''#############################################################################
#                                     MAIN                                     #
#############################################################################'''
if __name__ == '__main__':
    argv = sys.argv
    
    
    # Generate processes
    processes = RandomProcessFactory(argv).generate()
    # Queue processes
    sjf = SJFScheduler(deepcopy(processes))
    srt = SRTScheduler(deepcopy(processes))
    fcfs = FCFSScheduler(deepcopy(processes))
    rr = RRScheduler(deepcopy(processes))
    # Execute processes
    sjf.execute()
    srt.execute()
    fcfs.execute()
    rr.execute()
    # Output process logs
    print(sjf.logs())
    print(srt.logs())
    print(fcfs.logs())
    print(rr.logs())
    # Output statistics to file
    
