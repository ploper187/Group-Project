__author__ = "Cameron Scott, Adam Kuniholm, and Sam Fite"
__copyright__ = "Copyright 2019, Cameron Scott, Adam Kuniholm, and Sam Fite"
__credits__ = ["Cameron Scott", "Adam Kuniholm", "Sam Fite"]
__license__ = "MIT"

from enum import Enum
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


class ContextSwitch:
    timestamp = -1
    is_preemption = False
    old_process = None
    new_process = None

    def __init__(self, timestamp, old_process, new_process):
        self.timestamp = timestamp
        self.old_process = old_process
        self.new_process = new_process
        self.is_preemption = self.old_process.remaining_time > 0


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
    active_time = -1
    burst_time = -1
    state = State.READY

    def __init__(self, timestamp, burst_time):
        # TODO: Implement
        pass

    def time_remaining(self):
        return self.burst_time - self.active_time

    def is_completed(self):
        return self.time_remaining() > 0

    def preempted(self, timestamp):
        # TODO: Implement
        pass

    def burst(self, timestamp):
        # TODO: Implement
        pass

    def handle_context_switch(self, contextSwitch):
        assert(contextSwitch.new_process != contextSwitch.old_process)
        if (self == contextSwitch.new_process):
            self.burst(contextSwitch.timestamp)
        elif (self == contextSwitch.old_process):
            self.preempted(contextSwitch.timestamp)
        else:
            return


class Scheduler:
    queue = []
    context_switches = []  # Context switches
    cpu_burst_time = -1
    tot_wait_time = -1
    tot_turnaround_time = -1
    def execute(self):
        pass

    def __init__(self, processes):
        self.queue = list(filter(lambda x: not x.is_completed(), processes))

    def __str__(self):
        # TODO: Format
        n = len(self.queue)
        s =  '-- average CPU burst time: {0} ms\n'.format(self.cpu_burst_time/n)
        s += '-- average wait time: {0} ms\n'.format(self.tot_wait_time/n) 
        s += '-- average turnaround time: {0} ms\n'.format(self.tot_turnaround_time/n)
        s += '-- total number of context switches: {0}\n'.format(len(self.context_switches))
        s += '-- total number of preemptions: {0}\n'.format(len(filter(lambda cs: cs.is_preemption, self.context_switches)))
        return s

# Shorted Job First Process Scheduler


class SJFScheduler(Scheduler):
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        pass

# Shortest Time Remaining Process Scheduler


class SRTScheduler(Scheduler):
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        pass

# First Come First Server Process Scheduler


class FCFSScheduler(Scheduler):
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        pass

# Round Robin Process Scheduler


class RRScheduler(Scheduler):
    def execute(self):
        # SJF Algorithm
        # TODO: Implement
        pass


'''#############################################################################
#                                     MAIN                                     #
#############################################################################'''
if __name__ == '__main__':
    argv = sys.argv
    ARRIVAL_TIME_SEED = argv[1]
    LAMBDA = argv[2]
    ARRIVAL_TIME_CEIL = argv[3]
    NUM_PROCESSES = 4
    CONTEXT_SWITCH_DURATION = argv[5]
    ALPHA = argv[6]
    TIME_SLICE = argv[7]
    RR_SHOULD_PUSH_FRONT = True if argv[8] == 'BEGINNING' else False
    # TODO: Generate processes
    processes = []

    # Queue, execute processes
    sjf = SJFScheduler(processes)
    sjf.execute()
    srt = SRTScheduler(processes)
    srt.execute()
    fcfs = FCFSScheduler(processes)
    fcfs.execute()
    rr = RRScheduler(processes)
    rr.execute()

    print(sjf)
    print(srt)
    print(fcfs)
    print(rr)
    # TODO: Output schedule log to file
    
