# include<io
1. Semaphore

Definition:
A semaphore is a synchronization tool used to control access to shared resources in a multi-process system.

Types of Semaphore

Binary Semaphore (Mutex)

Value = 0 or 1

Used for mutual exclusion

Counting Semaphore

Value ≥ 0

Used when multiple resources are available

Basic Operations

wait() / P() → decrease value

signal() / V() → increase value

Simple Idea:

wait() → request resource

signal() → release resource

2. Producer–Consumer Problem (Using Semaphore)

Definition:
A classic synchronization problem where:

Producer → produces data

Consumer → consumes data

Both share a common buffer

Problem

Producer may overflow buffer

Consumer may read empty buffer

Solution Using Semaphores

We use 3 semaphores:

mutex = 1 → for mutual exclusion

empty = N → number of empty slots

full = 0 → number of filled slots

Working (Easy Steps)
Producer

wait(empty) → check space

wait(mutex) → lock buffer

Add item

signal(mutex) → unlock

signal(full) → increase filled slots

Consumer

wait(full) → check data available

wait(mutex) → lock buffer

Remove item

signal(mutex) → unlock

signal(empty) → increase empty slots

Simple Example

Buffer size = 5

Producer adds items → empty decreases, full increases

Consumer removes items → full decreases, empty increases

Short Exam Version (3–4 lines)

A semaphore is a synchronization mechanism used to control access to shared resources using wait() and signal() operations. In the Producer–Consumer problem, semaphores like mutex, empty, and full are used to avoid race conditions, buffer overflow, and underflow. This ensures proper coordination between producer and consumer.