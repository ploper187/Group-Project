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

        new_list = sorted(q, key=lambda x: (x.tau, x.name))

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
    def __str__(self): return " ".join([str(self.process), "[NEW] (arrival time", str(
        self.process.creation_ts), "ms)", str(self.process.bursts_remaining), "CPU bursts"])


class ProcessArrival(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process),
        "(tau", str(self.process.tau) + "ms)",
            "arrived; added to ready queue", self.queue()])


class CPUBurstBegun(ProcessEvent):
    burst_time = -1

    def __init__(self, simulation, process, burst_time):
        self.simulation = (simulation)
        self.process = (process)
        self.burst_time = burst_time

    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process),
                "started using the CPU for", str(self.burst_time) + "ms", "burst", self.queue()])


class CPUBurstEnded(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process),
                "completed a CPU burst;", str(self.process.bursts_remaining), "bursts to go", self.queue()])


class ProcessTauRecalculated(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", "Recalculated tau =", str(
        self.process.tau) + "ms", "for process", self.process.name, self.queue()])


class Preemption(ProcessEvent):
    new_process = None

    def __init__(self, simulation, old_process, new_process):
        self.simulation = (simulation)
        self.process = (old_process)
        self.new_process = (new_process)

    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.new_process),
                              "(tau", str(self.new_process.tau) +
                                "ms) completed I/O and will preempt",
                              self.process.name, self.queue()])


class Q_IOBurstStarts(ProcessEvent):
    io_burst_time = -1

    def __init__(self, simulation, process, io_burst_time):
        self.simulation = (simulation)
        self.process = (process)
        self.io_burst_time = io_burst_time

    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process), "switching out of CPU; will block on I/O until time", str(
        self.simulation.current_time + self.io_burst_time + int(self.process.context_switch_duration / 2)) + "ms", self.queue()])


class IOBurstEnds(ProcessEvent):
    def __str__(self): return " ".join([self.timestamp_str() + ":", str(self.process), "(tau", str(
        self.process.tau) + "ms)", "completed I/O; added to ready queue", self.queue()])


class ProcessCompleted(ProcessEvent):
    def __str__(self): return " ".join(
        [self.timestamp_str() + ":", str(self.process), "terminated", self.queue()])


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
    alpha = -1
    context_switch_duration = -1
    state = State.READY

    def __init__(self, name, timestamp,
                 num_bursts, cpu_burst_times, io_burst_times, alpha, context_switch_duration, tau=0):
        self.name = name
        self.tau = math.ceil(tau)
        self.alpha = alpha
        self.events = []
        self.creation_ts = timestamp
        self.state = Process.State.READY
        self.burst_times = cpu_burst_times
        self.io_burst_times = io_burst_times
        self.bursts_remaining = len(cpu_burst_times)
        self.context_switch_duration = context_switch_duration

    def time_remaining(self):
        return sum(self.burst_times[0:self.burst_index]) + sum(self.io_burst_times[0:self.io_burst_index])

    def is_completed(self):
        return self.burst_index == len(self.burst_times)

    def avg_cpu_burst_time(self):
        return mean(self.burst_times)

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


    def execute(self): pass

    def __init__(self, processes):
        self.queue = processes
        self.num_processes = len(self.queue)
        self.post_ready = []
        self.post_cpu = []
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
    def avg_cpu_burst_time(self):
        ps = self.completed # All processes
        n = len(ps)
        burst_times = [p.avg_cpu_burst_time() for p in ps]
        avg_burst_time = sum(burst_times)/max(n, 1)

        return avg_burst_time
    '''

    Turnaround times are to be measured for each process that you simulate. 
         - End-to-end time a process spends in executing a single CPU burst.
         [________________________1 Turnaround Time__________________________]
         * ------------------ . ------------------!----!----!--------------- * 
         |                    |                   \    |    /                |
         Arrival              |                   Preemptions                | 
                              Switched in                           Switch out          
        Event on arrival, event on completion of CPU burst             
    '''
    def avg_cpu_burst_turnaround_time(self):
        ps = self.completed # All processes
        # n = len(ps)
        turnarounds = []
        for p in ps:
            turnaround = p.completion_ts - p.creation_ts 
            # print(self.name, "creation cs:", p.creation_ts, "end_cs:", p.completion_ts)
            # turnaround = 0
            # for e in p.events:
            #     if (isinstance(e, CPUBurstBegun)):
            #         start_time = e.timestamp
            #     elif (isinstance(e, CPUBurstEnded) or isinstance(e, Q_IOBurstStarts)):
            #         turnaround += e.timestamp-start_time
            #         start_time = 0
            turnarounds.append(turnaround/len(p.burst_times))                
        return mean(turnarounds) if len(turnarounds) > 0 else 0

    def avg_cpu_wait_time(self):
        ps = self.completed # All processes
        n = len(ps)
        waits = []
        for p in ps:
            wait = p.completion_ts - p.creation_ts - sum(p.burst_times) - sum(p.io_burst_times) - p.context_switch_duration*len(p.burst_times)
            waits.append(wait/len(p.burst_times))
    
        return mean(waits) if len(waits) > 0 else 0
    '''
        Also note that to count the number of context switches, 
        you should count the number of times a process starts using the CPU.
    '''
    def num_context_switches(self):
        ps = self.completed # All processes
        n = len(ps)
        # BUG: This is not always true w/ preemptions
        cs = [len(p.burst_times) for p in ps]

        return sum(cs)
    
    def num_preemptions(self):
        return len([e for e in self.events if isinstance(e, Preemption)])


    def __str__(self):
        return '\n'.join(\
            ['Algorithm {0}'.format(self.name),   \
             '-- average CPU burst time: {0:0.3f} ms'.format(self.avg_cpu_burst_time()), \
             '-- average wait time: {0:0.3f} ms'.format(self.avg_cpu_wait_time()), \
             '-- average turnaround time: {0:0.3f} ms'.format(self.avg_cpu_burst_turnaround_time()), \
             '-- total number of context switches: {0:0.3f}'.format(self.num_context_switches()), \
             '-- total number of preemptions: {0:0.3f}\n'.format(self.num_preemptions())])
 
    def is_completed(self): return len(self.completed) == self.num_processes
    def logs(self):
        ret = sorted(self.log, key = lambda x: (x[1], x[2]))
        string = ""
        for event in ret:
            string += event[0] + "\n"
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
    def begin(self):
        self.log_event(StartSimulation(self))
    def end(self):
        self.log_event(EndSimulation(self))
    def process_arrived(self, process):
        # TODO: Configure process arrival
        self.log_event(ProcessArrival(self, process))
    def process_burst(self, process):
        # TODO: Configure process CPU burst begins
        self.log_event(CPUBurstBegun(self, process, process.burst_times[process.burst_index]))
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
    def process_terminated(self, process):
        # TODO: Configure process termination
        process.completion_ts = self.current_time
        self.log_event(ProcessCompleted(self, process))

''' Shorted Job First Process Scheduler '''
class SJFScheduler(Scheduler):
    name = "SJF"
    def execute(self):
        # SJF Algorithm
        self.begin()

        done = False
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
                        self.process_terminated(process)
                    else:
                        # The process that just finished its burst needs to recalculate tau and start I/O
                        process.tau = math.ceil(process.alpha * process.burst_times[process.burst_index - 1] + (1 - process.alpha) * process.tau)
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

        # self.process_arrived(self.queue[0])

        # self.process_burst(self.queue[0])
        # self.process_ends_burst(self.queue[0])

        # self.process_io_burst(self.queue[0])
        # self.process_ends_io_burst(self.queue[0])

        # self.recaculated_tau(self.queue[0])

        # self.process_terminated(self.queue[0])

        self.end()


''' First Come First Server Process Scheduler '''
class FCFSScheduler(Scheduler):
    name = "FCFS"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        self.begin()

        # self.process_arrived(self.queue[0])

        # self.process_burst(self.queue[0])
        # self.process_ends_burst(self.queue[0])

        # self.process_io_burst(self.queue[0])
        # self.process_ends_io_burst(self.queue[0])

        # self.recaculated_tau(self.queue[0])

        # self.process_terminated(self.queue[0])

        self.end()

''' Round Robin Process Scheduler '''
class RRScheduler(Scheduler):
    name = "RR"
    def execute(self):
        # SJF Algorithm
        # TODO: Implement

        ''' 
        Example of Scheduler use
        Subclass for custom behavior 
        '''
        self.begin()
        
        # self.process_arrived(self.queue[0])

        # self.process_burst(self.queue[0])
        # self.process_ends_burst(self.queue[0])

        # self.process_io_burst(self.queue[0])
        # self.process_ends_io_burst(self.queue[0])

        # self.recaculated_tau(self.queue[0])

        # self.process_terminated(self.queue[0])

        self.end()


'''#############################################################################
#                                     MAIN                                     #
# '''
if __name__ == '__main__':
    is_debug = len(sys.argv) > 0 and sys.argv[-1] == "debug"
    # Generate processes
    processes = RandomProcessFactory(sys.argv).generate()
    
    # Queue processes
    sjf = SJFScheduler(copy(processes))   # Shortest Job First
    srt = SRTScheduler(copy(processes))   # Shortest Time Remaining 
    fcfs = FCFSScheduler(copy(processes)) # First Come First Serve
    rr = RRScheduler(copy(processes))     # Round Robin
    
    # Execute (create) Schedules
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

    # Done
    if (is_debug):
        print(str(sjf) + str(srt) + str(fcfs) + str(rr))
