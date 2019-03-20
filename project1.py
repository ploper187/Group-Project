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
class SimulationEvent(Event):
    simulation = None
    action = ""
    def __init__(self, simulation): self.simulation = simulation
    def __str__(self): 
        s = "time " + str(self.simulation.current_time) + "ms: "
        s += "Simulator " + self.action + " for " +  self.simulation.name
        s += " [Q "
        if (len(self.simulation.ready) > 0):
                s += " ".join(str(p) for p in self.simulation.ready)
        else:   s += "<empty>"
        s += "]"
        return s
class StartSimulation(SimulationEvent): action = "started"
class EndSimulation(SimulationEvent):   action = "ended"
class ProcessEvent(Event):
    process = None
    simulation = None
    def timestamp(self): 
        return "time " + str(self.simulation.current_time) + "ms"
    def queue(self): 
        q = self.simulation.ready
        s = "[ Q "
        if (len(q) > 0):
                s += " ".join(str(p) for p in q)
        else:   s += "<empty>"
        s += "]"
        return s
    def __init__(self, simulation, process):
        self.process = process
        self.simulation = simulation
class NewProcess(ProcessEvent):
    def __str__(self): return " ".join([self.process.name, "[NEW] (arrival time", str(self.process.creation_ts), "ms)", str(self.process.bursts_remaining), "CPU bursts"])
class ProcessArrival(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp() + ":", str(self.process), "(tau 1000ms) arrived; added to ready queue", self.queue()])
class CPUBurstBegun(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp() + ":", str(self.process), "started using the CPU for 505ms burst", self.queue()])
class CPUBurstEnded(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp() + ": ", str(self.process), "completed a CPU burst; 88 bursts to go", self.queue()])
class ProcessTauRecalculated(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp() + ":", "Recalculated tau = 753ms for process", self.process.name, self.queue()])
class Preemption(ProcessEvent):
    new_process = None
    def __init__(self, simulation, old_process, new_process):
        self.simulation = simulation
        self.process = old_process
        self.new_process = new_process
    def __str__(self): return " ".join([self.timestamp() + ":", str(self.new_process),\
                              "(tau", str(self.new_process.tau) + "ms) completed I/O and will preempt",\
                              self.process.name, self.queue()])
class IOBurstStarts(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp() + ": ", str(self.process), "switching out of CPU; will block on I/O until time 105805ms", self.queue()])
class IOBurstEnds(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp() + ":", str(self.process), "(tau 1104ms) completed I/O; added to ready queue", self.queue()])
class ProcessCompleted(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp() + ":", str(self.process), "terminated", self.queue()])

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
    tau = -1
    state = State.READY

    def __init__(self, name, timestamp, 
                 num_bursts, cpu_burst_time, io_burst_time, tau = 0):
        self.name = name
        self.tau = tau
        self.creation_ts = timestamp
        self.bursts_remaining = num_bursts
        self.state = Process.State.READY
        self.cpu_burst_time = cpu_burst_time

        self.io_burst_time = io_burst_time

    def time_remaining(self):
        return self.bursts_remaining*(self.cpu_burst_time + self.io_burst_time)

    def is_completed(self):
        return self.bursts_remaining > 0

    def __str__(self): return "Process " + self.name


class Scheduler:
    name = "None"
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
    # Queue
    queue = []

    def execute(self): pass
    def __init__(self, processes): 
        self.queue = processes
        self.ready = []
        self.events = []
        self.current_time = 0
        self.cpu_burst_time = 0
        self.tot_turnaround_time = 0
        self.tot_wait_time = 0
        self.completed = []
        self.blocked = []
        self.active = None
        for process in self.queue:
            self.log_event(NewProcess(self, process))

    def __str__(self):
        n = len(self.completed) + len(self.queue) + len(self.ready)
        return '\n'.join(\
            ['Algorithm {0}'.format(self.name),   \
             '-- average CPU burst time: {0} ms'.format(self.cpu_burst_time/n), \
             '-- average wait time: {0} ms'.format(self.tot_wait_time/n), \
             '-- average turnaround time: {0} ms'.format(self.tot_turnaround_time/n), \
             '-- total number of context switches: {0}'.format(len(list(filter(lambda cs: cs is Preemption, self.events)))), \
             '-- total number of preemptions: {0}\n'.format(len(list(filter(lambda cs: cs is Preemption, self.events))))])
 
    def is_completed(self): return len(self.queue) == 0
    def logs(self):
        return '\n'.join(str(e) for e in self.events) + '\n'
    '''
        Log an event, such as a preemption, process completion, or context switch

        e.g. self.log_event(ProcessCompleted(p1))
    '''
    def log_event(self, event):
        event.timestamp = self.current_time
        event.queue = deepcopy(self.ready)
        self.events.append(event)
    
    def begin(self):
        self.log_event(StartSimulation(self))
    def end(self):
        self.log_event(EndSimulation(self))
    def process_arrived(self, process):
        self.log_event(ProcessArrival(self, process))
    def process_burst(self, process):
        self.log_event(CPUBurstBegun(self, process))
    def process_ends_burst(self, process):
        self.log_event(CPUBurstEnded(self, process))
    def recaculated_tau(self, process):
        self.log_event(ProcessTauRecalculated(self, process))
    def process_io_burst(self, process):
        self.log_event(IOBurstStarts(self, process))
    def process_ends_io_burst(self, process):
        self.log_event(IOBurstEnds(self, process))
    
''' Shorted Job First Process Scheduler '''
class SJFScheduler(Scheduler):
    name = "SJF"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        self.begin()


        self.end()


'''Shortest Time Remaining Process Scheduler'''
class SRTScheduler(Scheduler):
    name = "SRT"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        self.begin()


        self.end()


''' First Come First Server Process Scheduler '''
class FCFSScheduler(Scheduler):
    name = "FCFS"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        self.begin()


        self.end()

''' Round Robin Process Scheduler '''
class RRScheduler(Scheduler):
    name = "RR"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        self.begin()


        self.end()

class ProcessFactory:
    def generate(self):
        return []

class RandomProcessFactory(ProcessFactory):
    ics = None
    def __init__(self, argv):
        self.ics = RandomProcessFactory.InitialConditions(argv)
    
    # Uncomment the next line if using Python 2.x...
# from __future__ import division
    class Rand48(object):
        def __init__(self, seed):
            self.n = seed
        def seed(self, seed):
            self.n = seed
        def srand(self, seed):
            self.n = (seed << 16) + 0x330e
        def next(self):
            self.n = (25214903917 * self.n + 11) & (2**48 - 1)
            return self.n
        def drand(self):
            return self.next() / 2**48
        def lrand(self):
            return self.next() >> 17
        def mrand(self):
            n = self.next() >> 16
            if n & (1 << 31):
                n -= 1 << 32
            return n   
    def generate(self):
        r = RandomProcessFactory.Rand48(0)
        r.srand(self.ics.seed)
        processes = []
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(self.ics.num_processes):
            arrival_time = -1
            while (arrival_time < 0 or arrival_time > self.ics.arrival_time_cieling):
                arrival_time = math.floor(self.ics.map_to_exp(r.drand()))
            num_bursts = math.trunc(r.drand()*100.0)+1            
            burst_times = []
            io_burst_times = []
            for j in range(num_bursts): 
                burst_times.append(r.drand())
                if (j < num_bursts - 1):
                    io_burst_times.append(r.drand())
            process = Process(letters[i], arrival_time, num_bursts, burst_times, io_burst_times)
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
            return (-1)*math.log(float(uniform_value))/self.lambda_value

        def __init__(self, argv):
            self.seed = int(argv[1])
            self.lambda_value = float(argv[2])
            self.arrival_time_cieling = float(argv[3])
            self.num_processes = int(argv[4])
            self.context_switch_duration = float(argv[5])
            self.alpha = float(argv[6])
            self.time_slice = float(argv[7])
            self.rr_should_push_front = True if len(argv) == 9 and argv[8] == 'BEGINNING' else False


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
    o = open("simout.txt", "w+")
    o.write(str(sjf))
    o.write(str(srt))
    o.write(str(fcfs))
    o.write(str(rr))
    o.close()

