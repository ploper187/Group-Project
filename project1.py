__author__ = "Cameron Scott, Adam Kuniholm, and Sam Fite"
__right__ = "right 2019, Cameron Scott, Adam Kuniholm, and Sam Fite"
__credits__ = ["Cameron Scott", "Adam Kuniholm", "Sam Fite"]
__license__ = "MIT"


from statistics import mean
import sys
import random
import math
from copy import copy
from enum import Enum
import time
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
    def __init__(self, simulation): self.simulation = (simulation)

    def __str__(self):
        s = "time " + str(self.simulation.current_time) + "ms: "
        s += "Simulator " + self.action + " for " + self.simulation.name
        s += " [Q "
        if (len(self.simulation.ready) > 0):
                s += " ".join(p.name for p in self.simulation.ready)
        else:   s += "<empty>"
        s += "]"
        return s


class StartSimulation(SimulationEvent): action = "started"


class EndSimulation(SimulationEvent):   action = "ended"


class ProcessEvent(Event):
    process = None
    simulation = None
    def timestamp_str(self): return "time " + str(self.timestamp) + "ms"

    def queue(self):
        q = self.simulation.ready

        new_list = sorted(q, key=lambda x: (x.running_tau, x.name))

        s = "[Q "
        if (len(q) > 0):
                s += " ".join(p.name for p in new_list)
        else:   s += "<empty>"
        s += "]"
        return s

    def queue2(self, q):
        new_list = q
        s = "[Q "
        if (len(q) > 0):
                s += " ".join(p.name for p in new_list)
        else:   s += "<empty>"
        s += "]"
        return s

    def __init__(self, simulation, process):
        self.process = (process)
        self.simulation = (simulation)


class NewProcess(ProcessEvent):
    def __str__(self): return "".join([str(self.process), " [NEW] (arrival time ", str(
        self.process.creation_ts),  " ms) ", str(self.process.bursts_remaining), " CPU burst", ("s" if self.process.bursts_remaining > 1 else "")])


class ProcessArrival(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process),
        "(tau", str(self.process.tau) + "ms)",
            "arrived; added to ready queue", self.queue()])

class ProcessArrivalSam(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process),
            "arrived; added to ready queue", self.queue2(self.simulation.ready)])

class CPUBurstBegun(ProcessEvent):
    burst_time = -1

    def __init__(self, simulation, process, burst_time):
        self.simulation = (simulation)
        self.process = (process)
        self.burst_time = burst_time

    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process),
                "started using the CPU for", str(int(self.burst_time)) + "ms", "burst", (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])

class CPUBurstContinued(ProcessEvent):
    burst_time = -1

    def __init__(self, simulation, process, burst_time):
        self.simulation = (simulation)
        self.process = (process)
        self.burst_time = burst_time

    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process),
                "started using the CPU with", str(int(self.burst_time)) + "ms", "remaining", (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])

class CPUBurstEnded(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process),
                "completed a CPU burst;", str(self.process.bursts_remaining), "bursts to go", (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])

class ProcessTauRecalculated(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", "Recalculated tau =", str(
        self.process.tau) + "ms", "for process", self.process.name, (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])

# CAM
class PreemptionSam(ProcessEvent):
    new_process = None
    time_left = None

    def __init__(self, simulation, old_process, new_process, time_left):
        self.simulation = (simulation)
        self.process = (old_process)
        self.new_process = (new_process)
        self.time_left = time_left

    def __str__(self): 
        if (self.process.name == self.new_process.name):
            return " ".join([self.timestamp_str() + \
                    ": Time slice expired; no preemption because ready queue is empty", 
                                  self.queue2(self.simulation.ready)])
        else:
            return "".join([self.timestamp_str() + ": Time slice expired; process ", 
                                  str(self.process.name),
                                    " preempted with ", str(self.time_left), "ms to go "
                                  ,self.queue2(self.simulation.ready)])

class Preemption(ProcessEvent):
    new_process = None

    def __init__(self, simulation, old_process, new_process):
        self.simulation = (simulation)
        self.process = (old_process)
        self.new_process = (new_process)

    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.new_process),
                              "(tau", str(self.new_process.tau) +
                                "ms) completed I/O and will preempt",
                              self.process.name, (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])

class ImmediatePreemption(ProcessEvent):
    new_process = None

    def __init__(self, simulation, old_process, new_process):
        self.simulation = (simulation)
        self.process = (old_process)
        self.new_process = (new_process)

    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.new_process),
                              "(tau", str(self.new_process.tau) +
                                "ms) will preempt",
                              self.process.name, (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])

class Q_IOBurstStarts(ProcessEvent):
    io_burst_time = -1

    def __init__(self, simulation, process, io_burst_time):
        self.simulation = (simulation)
        self.process = (process)
        self.io_burst_time = io_burst_time

    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process), "switching out of CPU; will block on I/O until time", str(
        self.simulation.current_time + self.io_burst_time + int(self.process.context_switch_duration / 2)) + "ms", (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])


class IOBurstEnds(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process), "(tau", str(
        self.process.tau) + "ms)", "completed I/O; added to ready queue", (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])

class IOBurstEndsSam(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process), "completed I/O; added to ready queue", self.queue2(self.simulation.ready)])

class ProcessCompleted(ProcessEvent):
    def __str__(self): return " ".join(
        [self.timestamp_str() + ":", str(self.process), "terminated", (self.queue() if not self.simulation.sam else self.queue2(self.simulation.ready))])


class Process:
    class State(Enum):
        RUNNING = 0
        READY = 1
        BLOCKED = 2
        COMPLETED = -1
    name = None
    events = []
    creation_ts = -1
    ready_ts = -1
    completion_ts = -1
    last_active_ts = -1
    last_blocked_ts = -1
    bursts_remaining = -1
    burst_index = 0
    burst_times = []
    io_burst_index = 0
    io_burst_times = []
    tau = -1
    running_tau = -1
    alpha = -1
    context_switch_duration = -1
    state = State.READY
    status = "fresh"
    last_burst = -1
    wait = 0
    num_times_waited = 0

    def __init__(self, name, timestamp,
                 num_bursts, cpu_burst_times, io_burst_times, alpha, context_switch_duration, tau=0):
        self.name = name
        self.tau = math.ceil(tau)
        self.running_tau = self.tau
        self.alpha = alpha
        self.events = []
        self.creation_ts = timestamp
        self.state = Process.State.READY
        self.burst_times = cpu_burst_times
        self.io_burst_times = io_burst_times
        self.bursts_remaining = len(cpu_burst_times)
        self.context_switch_duration = context_switch_duration
        self.average_burst_time = mean(self.burst_times)
        self.total_burst_time = sum(self.burst_times)
        self.total_io_burst_time = sum(self.io_burst_times)
        

    def time_remaining(self):
        return sum(self.burst_times[0:self.burst_index]) + sum(self.io_burst_times[0:self.io_burst_index])

    def is_completed(self):
        return self.burst_index == len(self.burst_times)

    def avg_cpu_burst_time(self):
        return self.average_burst_time

    def __str__(self): return "Process " + self.name


class ProcessFactory:
    def generate(self): return []


class RandomProcessFactory(ProcessFactory):
    ics = None

    def __init__(self, argv):
        self.ics = RandomProcessFactory.InitialConditions(argv)

    # Uncomment the next line if using Python 2.x...
    # from __future__ import division
    class Rand48(object):
        def __init__(self, seed): self.n = seed
        def seed(self, seed):     self.n = seed
        def srand(self, seed):    self.n = (seed << 16) + 0x330e

        def next(self):
            self.n = (25214903917 * self.n + 11) & (2**48 - 1)
            return self.n

        def drand(self): return self.next() / 2**48
        def lrand(self): return self.next() >> 17

        def mrand(self):
            n = self.next() >> 16
            if n & (1 << 31):
                n -= 1 << 32
            return n

    def generate(self):
        # Pseudo-random numbers and predictability
        r = RandomProcessFactory.Rand48(0)
        r.srand(self.ics.seed)
        processes = []
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(self.ics.num_processes):
            arrival_time = -1
            # 1. Keep generating arrival times until a valid arrival time is generated
            while (arrival_time < 0 or arrival_time > self.ics.random_number_cieling):
                arrival_time = math.floor(self.ics.map_to_exp(r.drand()))
            # 2. num_bursts in [1, 100]
            num_bursts = math.trunc(r.drand()*100.0)+1
            burst_times = []
            io_burst_times = []
            # Generate burst times for each burst
            for j in range(num_bursts):
                burst_time = -1
                while (burst_time < 0 or burst_time > self.ics.random_number_cieling):
                    burst_time = math.ceil(self.ics.map_to_exp(r.drand()))
                burst_times.append(burst_time)
                io_burst_time = -1
                # The last burst doesn't use IO
                if (j < num_bursts - 1):
                    while (io_burst_time < 0 or io_burst_time > self.ics.random_number_cieling):
                        io_burst_time = math.ceil(
                            self.ics.map_to_exp(r.drand()))
                    io_burst_times.append(io_burst_time)

            initial_tau = 1 / self.ics.lambda_value
            process = Process(letters[i], arrival_time, num_bursts, burst_times,
                              io_burst_times, self.ics.alpha, self.ics.context_switch_duration, initial_tau)
            processes.append(process)
        return processes

    class InitialConditions:
        seed = -1
        lambda_value = -1
        random_number_cieling = -1
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
            self.random_number_cieling = float(argv[3])
            self.num_processes = int(argv[4])
            self.context_switch_duration = float(argv[5])
            self.alpha = float(argv[6])
            self.time_slice = float(argv[7])
            self.rr_should_push_front = True if len(
                argv) == 9 and argv[8] == 'BEGINNING' else False


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
    # Logs
    log = []
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
    # Number of processes that the schedule started with
    processes = 0

    sam = False

    # Stats
    avg_cpu_burst_time = 0
    avg_cpu_wait_time = 0
    avg_turnaround_time = 0
    num_context_switches = 0
    num_preemptions = 0

    def execute(self): pass

    def __init__(self, processes):
        self.queue = processes
        self.num_processes = len(self.queue)
        self.post_ready = None
        self.post_cpu = None
        self.ready = []
        self.log = []
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

    
    '''
    CPU burst time is defined as the amount of time a process is actually using the CPU. 
        - This measure does not include context switch times.
    '''
    # def avg_cpu_burst_time(self):
    #     ps = self.completed # All processes
    #     n = len(ps)
    #     burst_times = [p.avg_cpu_burst_time() for p in ps]
    #     avg_burst_time = sum(burst_times)/max(n, 1)
    #     return avg_burst_time
    # '''
    # Turnaround times are to be measured for each process that you simulate. 
    #      - End-to-end time a process spends in executing a single CPU burst.
         
    #      [________________________1 Turnaround Time__________________________]
    #      * ------------------ . ------------------!----!----!--------------- * 
    #      |                    |                   \    |    /                |
    #      Arrival              |                   Preemptions                | 
    #                           Switched in                           Switch out          
    #     Event on arrival, event on completion of CPU burst             
    # '''
    # def avg_cpu_burst_turnaround_time(self):
    #     ps = self.completed # All processes

    #     tot_turnaround = sum([p.completion_ts - p.creation_ts for p in ps])
    #     num_bursts = sum([len(p.burst_times) for p in ps])

    #     return tot_turnaround/max(num_bursts, 1)

    # def avg_cpu_wait_time(self):
    #     ps = self.completed # All processes
    #     es = self.events
    #     # n = len(ps)
    #     tot_wait = 0
    #     tot_wait += sum([p.completion_ts - p.creation_ts for p in ps])
    #     tot_wait -= sum([p.total_burst_time + p.total_io_burst_time  for p in ps])
    #     tot_wait -= sum([p.context_switch_duration*len(p.burst_times) for p in ps])
    #     start = 0
    #     for e in es:
    #         if (isinstance(e, ProcessArrival)):
    #             start = e.timestamp
            

    #     num_bursts = sum([len(p.burst_times) for p in ps])
    #     return tot_wait/max(num_bursts, 1)
    # '''
    #     Also note that to count the number of context switches, 
    #     you should count the number of times a process starts using the CPU.
    # '''
    # def num_context_switches(self):
    #     ps = self.completed # All processes
    #     # n = len(ps)
    #     # BUG: This is not always true w/ preemptions
    #     cs = sum([len(p.burst_times) for p in ps])
    #     return cs
    
    # def num_preemptions(self):
    #     return len([e for e in self.events if isinstance(e, Preemption) or isinstance(e, ImmediatePreemption)])


    def __str__(self):
        # self.avg_turnaround_time = sum(process.turnaround for process in self.completed) / len(self.completed)
        print(self.completed[0].wait, self.completed[1].wait)

        self.avg_cpu_wait_time = sum([process.wait / process.num_times_waited for process in self.completed if process.num_times_waited is not 0]) / len(self.completed)
        return '\n'.join(\
            ['Algorithm {0}'.format(self.name),   \
             '-- average CPU burst time: {0:0.3f} ms'.format(float(self.avg_cpu_burst_time)), \
             '-- average wait time: {0:0.3f} ms'.format(float(self.avg_cpu_wait_time)), \
             '-- average turnaround time: {0:0.3f} ms'.format(float(self.avg_turnaround_time)), \
             '-- total number of context switches: {0}'.format(int(self.num_context_switches)), \
             '-- total number of preemptions: {0}\n'.format(int(self.num_preemptions))])
 
    def is_completed(self): return len(self.completed) == self.num_processes
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
    def process_burst(self, process):
        # TODO: Configure process CPU burst begins
        self.log_event(CPUBurstBegun(self, process, process.burst_times[process.burst_index]))
    def process_continued_burst(self, process):
        # TODO: Configure process CPU burst begins
        self.log_event(CPUBurstContinued(self, process, process.burst_times[process.burst_index]))
    def process_ends_burst(self, process):
        # TODO: Configure process CPU burst ends
        self.log_event(CPUBurstEnded(self, process))
    def recalculated_tau(self, process):
        # TODO: Configure process recalculates Tau 
        # Actually recalculate and assign Tau
        self.log_event(ProcessTauRecalculated(self, process))
    def process_io_burst(self, process):
        # TODO: Configure process IO bursts begin
        self.log_event(Q_IOBurstStarts(self, process, process.io_burst_times[process.io_burst_index]))
    def process_ends_io_burst(self, process):
        # TODO: Configure process IO bursts end
        self.log_event(IOBurstEnds(self, process))
    def process_preempted(self, process, new_process):
        self.log_event(Preemption(self, process, new_process))
    def process_immediate_preempted(self, process, new_process):
        self.log_event(ImmediatePreemption(self, process, new_process))
    def process_preempted_sam(self, process, new_process, time_left):
        self.log_event(PreemptionSam(self, process, new_process, time_left))
    def process_terminated(self, process):
        # TODO: Configure process termination
        process.completion_ts = self.current_time
        self.log_event(ProcessCompleted(self, process))
    def process_arrived_sam(self, process):
        # TODO: Configure process arrival
        self.log_event(ProcessArrivalSam(self, process))
    def process_ends_io_burst_sam(self, process):
        # TODO: Configure process IO bursts end
        self.log_event(IOBurstEndsSam(self, process))
    # CAM
    def process_preempted_sam(self, process, new_process, time_left):
        self.log_event(PreemptionSam(self, process, new_process, time_left))

''' Shorted Job First Process Scheduler '''
class SJFScheduler(Scheduler):
    name = "SJF"
    def execute(self):
        # SJF Algorithm
        self.begin()
        self.avg_turnaround_time = sum([sum([burst + process.context_switch_duration for burst in process.burst_times]) for process in self.queue]) / sum([len(process.burst_times) for process in self.queue])

        done = False
        self.post_cpu = []
        self.post_ready = []
        while(not self.is_completed()):

            to_be_removed = []
            for process, completion in self.post_ready:
                if completion == self.current_time:
                    if not self.current_time > 999:
                        self.process_burst(process)
                    process.burst_index += 1
                    process.bursts_remaining -= 1
                    to_be_removed.append((process, completion))

            # Remove any process from the post_ready list that is done with the context switch
            for process in to_be_removed:
                self.post_ready.remove(process)

            # Check for arrivals
            to_be_removed.clear()
            for process in self.queue:
                if process.creation_ts == self.current_time:
                    self.ready.append(process)
                    if not self.current_time > 999:
                        self.process_arrived(process)
                    to_be_removed.append(process)

            # Remove any process from the queue that arrived
            for process in to_be_removed:
                self.queue.remove(process)

            # Check if anything that is blocking is finished blocking
            to_be_removed.clear()
            for blocked_process in self.blocked:
                if blocked_process.last_blocked_ts == self.current_time:

                    if not blocked_process.is_completed():          # This is sort of unnecessary since it is checked before it enters I/O
                        self.ready.append(blocked_process)          # Should always enter the ready queue from this state
                    else:
                        blocked_process.state = blocked_process.State.COMPLETED
                        self.completed.append(blocked_process)
                    if not self.current_time > 999:
                        self.process_ends_io_burst(blocked_process)
                    to_be_removed.append(blocked_process)

            # Remove any process from the blocked list that arrived
            for process in to_be_removed:
                self.blocked.remove(process)

            to_be_removed.clear()
            for process, completion, destination in self.post_cpu:
                if completion == self.current_time:
                    
                    self.active = None # Release the CPU

                    if destination == 1:
                        self.blocked.append(process)
                    else:
                        self.completed.append(process)
                        if self.is_completed():
                            done = True
                    to_be_removed.append((process, completion, destination))

            if done:
                break

            # Remove any process from the post_ready list that is done with the context switch
            for process in to_be_removed:
                self.post_cpu.remove(process)

            # Check if the CPU is available yet
            if self.active == None:
                # Figure out which process in the ready queue has the lowest tau
                shortest_process = None

                new_list = sorted(self.ready, key = lambda x: (x.tau, x.name))

                if len(new_list) != 0:    # The ready queue could be empty
                    shortest_process = new_list[0]
                    shortest_process.state = shortest_process.State.RUNNING
                    self.active = shortest_process
                    shortest_process.last_active_ts = self.current_time + shortest_process.burst_times[shortest_process.burst_index] + (shortest_process.context_switch_duration / 2)
                    self.ready.remove(shortest_process)
                    self.post_ready.append((shortest_process, self.current_time + shortest_process.context_switch_duration / 2))
                    
            else:
                # Check if the running process is finished
                if self.active.last_active_ts == self.current_time:
                    process = self.active
                    if not self.current_time > 999:
                        self.process_ends_burst(process)    # Log the completion of this burst

                    # Check if this process has completed all of its bursts
                    if process.is_completed():
                        self.post_cpu.append((process, self.current_time + process.context_switch_duration / 2, 2))
                        process.state = process.State.COMPLETED
                        process.completion_ts = self.current_time
                        self.process_terminated(process)
                    else:
                        # The process that just finished its burst needs to recalculate tau and start I/O
                        process.tau = math.ceil(process.alpha * process.burst_times[process.burst_index - 1] + (1 - process.alpha) * process.tau)
                        process.running_tau = process.tau
                        if not self.current_time > 999:
                            self.recalculated_tau(process)

                        process.last_blocked_ts = self.current_time + process.io_burst_times[process.io_burst_index] + process.context_switch_duration / 2

                        process.state = process.State.BLOCKED
                        if not self.current_time > 999:
                            self.process_io_burst(process)
                        process.io_burst_index += 1
                        self.post_cpu.append((process, self.current_time + process.context_switch_duration / 2, 1))


            self.current_time += 1


        self.end()


'''Shortest Time Remaining Process Scheduler'''
class SRTScheduler(Scheduler):
    name = "SRT"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        self.begin()

        self.avg_cpu_burst_time = sum([sum(process.burst_times) for process in self.queue]) / sum([len(process.burst_times) for process in self.queue])
        self.avg_turnaround_time = sum([sum([burst + process.context_switch_duration for burst in process.burst_times]) for process in self.queue]) / sum([len(process.burst_times) for process in self.queue])
        
        
        while(not self.is_completed()):

            # ----------------------------------- Post Ready -> CPU -----------------------------------
            
            if self.post_ready != None:
                process, ready_time = self.post_ready
                if ready_time == self.current_time:
                    # The context switch is complete and we can move the process in to the CPU
                    process.last_active_ts = self.current_time + process.burst_times[process.burst_index]
                    self.active = process
                    if self.current_time <= 999:
                        if process.status == "preempted":
                            self.process_continued_burst(process)
                        else:
                            self.process_burst(process)
                    self.post_ready = None

            # ----------------------------------- New -> Ready -----------------------------------

            to_be_removed = []
            for process in self.queue:
                if process.creation_ts == self.current_time:
                    process.status = "fresh"
                    process.last_burst = process.burst_times[0]
                    self.ready.append(process)
                    if self.current_time <= 999:
                        self.process_arrived(process)
                    to_be_removed.append(process)

            for process in to_be_removed:
                self.queue.remove(process)

            # ----------------------------------- I/O -> Ready -----------------------------------

            to_be_removed.clear()
            for process in self.blocked:
                if process.last_blocked_ts == self.current_time:
                    # Done with I/O
                    process.status = "fresh"
                    self.ready.append(process)
                    process.num_times_waited += 1
                    if self.current_time <= 999:
                        if self.active is not None and self.active is not "Switching":
                            if process.tau >= self.active.running_tau:
                                self.process_ends_io_burst(process)
                        else:
                            self.process_ends_io_burst(process)
                    to_be_removed.append(process)

            for process in to_be_removed:
                self.blocked.remove(process)

            # ----------------------------------- Post CPU -> Ready-I/O-Done -----------------------------------
            
            if self.post_cpu != None:
                process, ready_time, destination = self.post_cpu
                if ready_time == self.current_time:
                    if destination == "completed":
                        self.completed.append(process)
                        if self.is_completed():
                            break
                    elif destination == "io":
                        process.last_burst = process.burst_times[process.burst_index]
                        process.last_blocked_ts = self.current_time + process.io_burst_times[process.io_burst_index]
                        process.io_burst_index += 1
                        self.blocked.append(process)
                    elif destination == "ready":
                        if process.tau is not process.running_tau:
                            process.status = "preempted"
                        else:
                            process.status = "fresh"
                        self.ready.append(process)
                        process.num_times_waited += 1


                    self.post_cpu = None
                    self.active = None  # Release the CPU

            # ----------------------------------- Ready -> Post Ready -----------------------------------
            
            '''
                A process should be sent to Post Ready if the CPU is available
            '''
            sorted_queue = sorted(self.ready, key=lambda x: (x.running_tau, x.name))
            if len(sorted_queue) != 0:  
                best_process = sorted_queue[0]
                if self.active is None:
                    if self.post_ready is None:
                        best_process.state = best_process.State.RUNNING
                        self.post_ready = (best_process, self.current_time + best_process.context_switch_duration / 2)
                        self.ready.remove(best_process)
                        self.num_context_switches += 1
                elif self.active != "Switching":
                    # Check for preemption (running tau is the estimated remaining time on the CPU)
                    if best_process.tau < self.active.running_tau:
                        self.num_preemptions += 1
                        # Need to remove process from CPU
                        if self.current_time <= 999:
                            if best_process.last_blocked_ts == self.current_time - 1:
                                self.process_immediate_preempted(self.active, best_process)
                            else:
                                self.process_preempted(self.active, best_process)
                        self.post_cpu = (self.active, self.current_time + self.active.context_switch_duration / 2, "ready")
                        self.active = "Switching"
            
            for process in self.ready:
                process.wait += 1

            # ----------------------------------- CPU -> Post CPU -----------------------------------

            if self.active != None and self.active != "Switching":
                if self.active.last_active_ts == self.current_time:
                    # This process is finished with the CPU
                    self.active.burst_index += 1
                    destination = ("completed" if self.active.is_completed() else "io")
                    self.active.tau = math.ceil(self.active.alpha * self.active.last_burst + (1 - self.active.alpha) * self.active.tau)
                    if self.current_time <= 999:
                        self.recalculated_tau(self.active)
                        if destination == "io":
                            self.process_io_burst(self.active)
                    self.active.running_tau = self.active.tau
                    self.post_cpu = (self.active, self.current_time + self.active.context_switch_duration / 2, destination)
                    self.active.bursts_remaining -= 1
                    if self.current_time <= 999:
                        self.process_ends_burst(self.active)
                    if destination == "completed":
                        self.active.completion_ts = self.current_time
                        self.process_terminated(self.active)
                    self.active = "Switching"
                else:
                    self.active.running_tau -= 1
                    self.active.burst_times[self.active.burst_index] -= 1

            # time.sleep(0.1)
            self.current_time += 1

        self.end()


''' First Come First Server Process Scheduler '''
class FCFSScheduler(Scheduler):
    name = "FCFS"
    def execute(self):
        self.sam = True
        self.begin()
        done = False
        self.post_cpu = []
        self.post_ready = []
        while ( not self.is_completed()):

            to_be_removed = []
            # handle post_cpu 
            for process, completion, destination in self.post_cpu:
                if completion == self.current_time:
                    # FREE the CPU
                    self.active = None
                    # if meant for IO
                    if destination == 1:
                        process.io_burst_index += 1
                        self.blocked.append(process)
                    # if completed
                    elif destination == 2:
                        self.completed.append(process)
                        process.completion_ts = self.current_time
                        if self.is_completed():
                            done = True
                    # if meant for ready queue
                    elif destination == 3:
                        process.num_times_waited += 1
                        if push_back_ready:
                            self.ready.append(process)
                        else:
                            self.ready.insert(0, process)
                    to_be_removed.append((process,  completion, destination) )

            # if the scheduler is done
            if done:
                break

            # remove any process from the post_cpu list that is done with context switch
            for process in to_be_removed:
                self.post_cpu.remove(process)


            # check for process that are ready to execute burst, start TIMESLICE
            to_be_removed.clear()
            for process, completion in self.post_ready:
                if completion == self.current_time:
                    if not self.current_time > 999:
                        self.process_burst(process)
                    process.burst_index += 1
                    process.bursts_remaining -= 1
                    to_be_removed.append((process, completion))
            
            # remove any process from the post_ready list that is done with the context switch
            for process in to_be_removed:
                self.post_ready.remove(process)



            # Check for arrivals
            to_be_removed.clear()
            for process in self.queue:
                if process.creation_ts == self.current_time:
                    process.num_times_waited += 1
                    self.ready.append(process)
                    if not self.current_time > 999:
                        self.process_arrived_sam(process)
                    to_be_removed.append(process)
            
            # remove any process from the queue that arrived
            for process in to_be_removed:
                self.queue.remove(process)


            
            # check if any blocked processes are done with IO
            to_be_removed.clear()
            for blocked_process in self.blocked:
                if blocked_process.last_blocked_ts == self.current_time:
                    if not blocked_process.is_completed():
                        blocked_process.num_times_waited += 1
                        self.ready.append(blocked_process)
                    else:
                        blocked_process.state = blocked_process.State.COMPLETED
                        self.completed.append(blocked_process)
                    if not self.current_time > 999:
                        self.process_ends_io_burst_sam(blocked_process)
                    to_be_removed.append(blocked_process)
            
            #remove any process from the blocked list that arrived
            for process in to_be_removed:
                self.blocked.remove(process)

            
            # CPU is available
            if self.active == None:
                if len(self.ready) > 0:
                    # get the next process in line
                    process = self.ready.pop(0)
                    process.state = process.State.RUNNING
                    self.active = process
                    process.last_active_ts = \
                            self.current_time + process.burst_times[process.burst_index] + (process.context_switch_duration / 2)
                    # append the process and the time it will be able to use the CPU
                    # we will log the cpu event later
                    self.post_ready.append((process, self.current_time+process.context_switch_duration /2))

            # CPU is not available
            elif self.active is not "Switching":
                # check if the currently running process is finished
                if self.active.last_active_ts == self.current_time:
                    process = self.active
                    if not self.current_time > 999 and not process.is_completed():
                        self.process_ends_burst(process) # LOG completion of burst

                    # check if this process has COMPLETED all of its bursts
                    if process.is_completed():
                        # append process, time when it can be switched out of cpu, and the number 2 for completed
                        self.post_cpu.append((process, self.current_time + process.context_switch_duration / 2, 2))
                        process.state = process.State.COMPLETED
                        process.completion_ts = self.current_time
                        self.process_terminated(process) # LOG process terminated
                    else:
                        process.last_blocked_ts = self.current_time + process.io_burst_times[process.io_burst_index] + process.context_switch_duration / 2
                        
                        process.state = process.State.BLOCKED
                        if not self.current_time > 999:
                            self.process_io_burst(process)
                        # append process, when it will be done with context switch, and the number 1 for IO
                        self.post_cpu.append((process, self.current_time + process.context_switch_duration / 2, 1))
                        self.active = "Switching"

            self.current_time += 1
 
        self.end()

''' Round Robin Process Scheduler '''
class RRScheduler(Scheduler):
    name = "RR"

    def execute(self, args):

        self.sam = True
        self.begin()
        time_slice = int(args[7])
        push_back_ready = (0 if len(args) == 9 and args[8] == 'BEGINNING' else 1)

        time_slice_counter = -1
        done = False

        self.post_cpu = []
        self.post_ready = []
        while ( not self.is_completed()):
            to_be_removed = []

            # handle post_cpu 
            for process, completion, destination in self.post_cpu:
                if completion == self.current_time:
                    # FREE the CPU
                    self.active = None
                    # if meant for IO
                    if destination == 1:
                        process.io_burst_index += 1
                        self.blocked.append(process)
                    # if completed
                    elif destination == 2:
                        self.completed.append(process)
                        process.completion_ts = self.current_time
                        if self.is_completed():
                            done = True
                    # if meant for ready queue
                    elif destination == 3:
                        # set time slice counter to -1 (waiting for a new process to start)
                        time_slice_counter = -1
                        if push_back_ready:
                            process.status = "preempted"
                            self.ready.append(process)
                            process.num_times_waited += 1
                            #print( 'time {0}ms: process {1} added to back of ready'.format(self.current_time, process) )
                        else:
                            self.ready.insert(0, process)
                            #print( 'process {0} added to front of read'.format(process) )
                    to_be_removed.append((process,  completion, destination) )

            # if the scheduler is done
            if done:
                break

            # remove any process from the post_cpu list that is done with context switch
            for process in to_be_removed:
                self.post_cpu.remove(process)



            # check for process that are ready to execute burst, start TIMESLICE
            to_be_removed.clear()
            for process, completion in self.post_ready:
                if completion == self.current_time:
                    if not self.current_time > 999:
                        if process.status == "fresh":
                            self.process_burst(process)
                        else:
                            self.process_continued_burst(process)
                    process.burst_index += 1
                    process.bursts_remaining -= 1
                    # reset time slice counter when we start a burst
                    time_slice_counter = 0
                    to_be_removed.append((process, completion))
            
            # remove any process from the post_ready list that is done with the context switch
            for process in to_be_removed:
                self.post_ready.remove(process)


            # Check for arrivals
            to_be_removed.clear()
            for process in self.queue:
                if process.creation_ts == self.current_time:
                    process.status = "fresh"
                    process.num_times_waited += 1
                    self.ready.append(process)
                    if not self.current_time > 999:
                        self.process_arrived_sam(process)
                    to_be_removed.append(process)
            
            # remove any process from the queue that arrived
            for process in to_be_removed:
                self.queue.remove(process)

            
            # check if any blocked processes are done with IO
            to_be_removed.clear()
            for blocked_process in self.blocked:
                if blocked_process.last_blocked_ts == self.current_time:
                    if not blocked_process.is_completed():
                        blocked_process.num_times_waited += 1
                        self.ready.append(blocked_process)
                    else:
                        blocked_process.state = blocked_process.State.COMPLETED
                        self.completed.append(blocked_process)
                    if not self.current_time > 999:
                        self.process_ends_io_burst_sam(blocked_process)
                    to_be_removed.append(blocked_process)
            
            #remove any process from the blocked list that arrived
            for process in to_be_removed:
                self.blocked.remove(process)

            
            
            # CPU is available
            if self.active == None and len(self.post_ready) == 0:
                if len(self.ready) > 0:
                    # get the next process in line
                    process = self.ready.pop(0)
                    process.state = process.State.RUNNING
                    self.active = process
                    # when the process will be done with everything
                    process.last_active_ts = \
                            self.current_time + process.burst_times[process.burst_index] + (process.context_switch_duration / 2)
                    # append the process and the time it will be able to use the CPU
                    # we will process the cpu usage event later
                    self.post_ready.append((process, self.current_time + process.context_switch_duration / 2))

            # CPU is not available
            elif self.active is not "Switching":
                # check if the currently running process is finished, timeslice is reset when burst is logged
                if self.active.last_active_ts == self.current_time:
                    process = self.active
                    time_slice_counter = -1
                    if not self.current_time > 999 and not process.is_completed():
                        self.process_ends_burst(process) # LOG completion of burst

                    # check if this process has COMPLETED all of its bursts
                    if process.is_completed():
                        # append process, time when it can be switched out of cpu, and the number 2 for completed
                        self.post_cpu.append((process, self.current_time + process.context_switch_duration / 2, 2))
                        process.state = process.State.COMPLETED
                        process.completion_ts = self.current_time
                        self.process_terminated(process) # LOG process terminated
                    else:
                        # the process that finished its burst has more bursts left and needs to start I/O
                        # BLOCKED
                        process.last_blocked_ts = self.current_time + process.io_burst_times[process.io_burst_index] + process.context_switch_duration / 2
                        process.state = process.State.BLOCKED
                        if not self.current_time > 999:
                            self.process_io_burst(process)
                        # append process, when it will be done with context switch, and the number 1 for IO
                        self.post_cpu.append((process, self.current_time + process.context_switch_duration / 2, 1))
                        self.active = "Switching"

                # if TIMESLICE is exceeded by a process
                elif time_slice_counter >= time_slice:
                    # if there are no processes to be fed into CPU, just log it
                    if (len(self.ready) == 0):
                        time_slice_counter = 0
                        if not self.current_time > 999:
                            # CAM
                            self.process_preempted_sam(self.active, self.active, 0)
                    # if we can PREEMPT 
                    else:
                        old_process = self.active
                        # since we increase the burst index and decrease burst remaining when a process
                        # enters the CPU, we need to revert the burst index and bursts remaining
                        old_process.burst_index -= 1
                        old_process.bursts_remaining += 1
                        # recalculate the reamaining burst time for the preempted process
                        old_process.burst_times[old_process.burst_index] = old_process.last_active_ts - self.current_time
                        old_process.state = old_process.State.READY
                        # put old process in post_cpu queue
                        self.post_cpu.append((old_process, self.current_time + (old_process.context_switch_duration / 2), 3) )
                        # reset time slice counter to wait until another process start to begin tslice
                        if not self.current_time > 999:
                            time_left = int(old_process.last_active_ts - self.current_time)
                            # CAM
                            self.process_preempted_sam(old_process, self.ready[0], time_left)
                        time_slice_counter = -1

            self.current_time += 1
            # time slice counter is negative when there is no process running
            if ( time_slice_counter >= 0 ):
                time_slice_counter += 1

        self.end()


'''#############################################################################
#                                     MAIN                                     #
# '''
if __name__ == '__main__':
    is_debug = len(sys.argv) > 0 and sys.argv[-1] == "debug"
    # Generate processes
    # processes = RandomProcessFactory(sys.argv).generate()
    
    # Queue processes
    sjf = SJFScheduler(RandomProcessFactory(sys.argv).generate())   # Shortest Job First
    srt = SRTScheduler(RandomProcessFactory(sys.argv).generate())   # Shortest Time Remaining 
    fcfs = FCFSScheduler(RandomProcessFactory(sys.argv).generate()) # First Come First Serve
    rr = RRScheduler(RandomProcessFactory(sys.argv).generate())     # Round Robin
    
    # Execute (create) Schedules
    sjf.execute()
    srt.execute()
    fcfs.execute()
    rr.execute(sys.argv)

    # Output process logs
    print(sjf.logs())
    print()
    print(srt.logs())
    print()
    print(fcfs.logs())
    print()
    print(rr.logs())

    # Output statistics to file
    o = open("simout.txt", "w+")
    o.write(str(sjf))
    o.write(str(srt))
    o.write(str(fcfs))
    o.write(str(rr))
    o.close()

    # Done
    if (is_debug):
        print(str(sjf) + str(srt) + str(fcfs) + str(rr))
