# QOA system requirements
## And the challenges of simulating greater than roughly 25 qubits in QOA on consumer hardware 

One of the major issues faced in current computing revolves around emulating quantum systems using
classical computers. This is due to the inherent architectural differences between quantum and
classical computing models. First and foremost, current computers are built with Boolean logic,
operating using transistors that have only two clearly distinguishable states, symbolized by the
binary digits 1 and 0. Quantum systems, however, are based on states of superposition, which require
mathematically complicated representations that grow exponentially with respect to their complexity.
The difficulty becomes increasingly severe once the number of qubits exceeds some level of critical
mass. Rather than being a function of processing capacity per se, the limiting factor is actually
the explosive growth of CPU cache requirements, memory allocations, and input/output operations
needed to service each cycle of computation.

To faithfully replicate quantum states and their corresponding qubit operations, both memory and
calculation capacity need to increase exponentially. Even though a simulation of around 25 qubits
would seemingly require a relatively modest amount of memory to accommodate (between 512 MB and 1
GB), attempts to exceed this level using standard consumer hardware often lead to severe system slowdowns
or crashes.

Eventually, a person will be subjected to resource requirements that are completely unrealistic,
leading to serious degradation of performance. The computation load becomes so heavy that often the
whole system becomes unstable, leading to crashes, freezes, or complete system failure.
This md file explains the justification behind the restrictions of using traditional hardware to
simulate quantum systems above a certain level, providing both mathematical and structural technical
explanations.

## Exponential Qubit Representation within QOA via classical methods like system memory, cache, and CPU

An examination of quantum simulation limitations requires consideration of the mathematical form of
quantum states in a classical computer's memory. A quantum system made up of n qubits is fully
described using a complex vector. There are exactly $2^n$ complex amplitudes in this vector with
each amplitude used to represent a specific basis state in the quantum system's associated Hilbert
space.

Exponential scaling is not so much a theoretical issue; it is an absolute mathematical limitation to
which we are bound. Adding each successive qubit to a quantum system leads to doubling of the state
vector's dimensions. This is because each successive qubit has the capability to enter a
superposition with all existing states and thus to produce an enormous set of possible quantum
states to be tracked at the same time, effectively doubling what had to be tracked before.
To put this in concrete numerical terms, a 25-qubit quantum system requires a state vector with
33,554,432 complex amplitudes ($2^{25}$). While this number might at first appear large, modern
computing systems with sufficient RAM can handle this sort of computational requirement. Storage of
the vectors usually takes up approximately 512 to 1024 megabytes of memory, depending on the
accuracy used in the representation of the complex numbers.

The memory needs are well matched to current capacities of RAM and can be efficiently handled by the
CPU's cache mechanisms provided the cache has adequate capacity to hold the entire state vector. The
key insight to take away is that so long as it is possible to include the entire quantum state
within the hierarchy of the CPU's cache environment, the simulation performance's effectiveness and speed is
acceptable.

However, the major change is revealed when considering the simulation of larger systems. For
instance, when simulating 28 qubits, the size of the state vector jumps to about 268 million complex
amplitudes. This is a shift from manageable memory requisites to considerable resource usage. Memory
needs jump to several tens of gigabytes, and the exponential growth quickly exhausts the memory
capacity of typical consumer-grade PCs.

Without workstation-class hardware featuring ample high-speed memory and a large CPU cache, it is
unlikely to run simulations involving numbers of qubits greater than 26 without suffering
considerable degradation of performance or system crashes.

Even in top-of-the-line consumer processors like the Ryzen 9 9950X, which represents some of the
most powerful consumer-level technology available, the machine begins to bog down at 27-qubit
simulations because of difficulties with accommodating the extreme input/output operations and
memory handling involved in each cycle of computation.

## Memory requirements and their practical limits

Within a quantum state vector, every complex amplitude requires an allocation of 16 bytes when using
double-precision floating-point representation. That allocation is split between 8 bytes for the
real part and 8 bytes for the imaginary part of the complex number. Such precision is typically
required to perform correct quantum simulations since the use of lower precision can result in
significant numerical errors that can accrue with the execution of a sequence of quantum operations.
To illustrate the idea of scalability, a simulation comprising 16 qubits demands around 1 megabyte
of memory space and is easily run on current computing hardware. The state vector can easily fit
into the typical sizes of CPU caches, which have access speeds ranging between 20 and 100 times that
of the latest DDR5 memory available. This speed difference is vital to the effectiveness of quantum
simulations since such software requires frequent access to and manipulation of the entire state
vector.

Outside the confines of the CPU cache, the demands for memory and input/output operations rise to
enormous and unsustainable levels. A 30 qubit state vector represents over a billion amplitudes
($2^{30}$), which takes about 16 gigabytes of RAM just to store in system memory. In contrast, a
system using 32 qubits would require about 64 gigabytes of memory.

These memory requirements do not include quantum gate computations' extra overheads, operational
activities' temporary storage requirements, and intrinsic operating system requirements for memory
management. Most personal computing devices are not capable of handling such large memory and
input/output requirements. Even sophisticated high-end workstations with substantial amounts of RAM
may also not be able to handle these requirements.

With increasing demand for memory space, the allocation of large contiguous blocks of memory is
becoming increasingly difficult. Memory fragmentation occurs when the operating system cannot find
sufficiently large contiguous blocks of memory; simultaneously, various restrictions placed by the
operating system might prevent effective memory allocation, even when sufficient total memory
theoretically exists.

## Why cpu cache is the real bottleneck

Quantum simulation effectiveness is not only determined by the amount of system RAM available but by
numerous other factors as well. Computational power overall is largely influenced by the
hierarchical structure of the CPU cache, which consists of finite amounts of very high-speed memory
(level 1, level 2, and level 3 cache) located right on the processor chip.

### why most cpus have limited cache

The CPU cache is built using static random-access memory (SRAM) technology. SRAM memory runs at much
higher speeds compared to dynamic random-access memory (DRAM), the main system memory, while this
comes with a very significant trade-off. Production of SRAM is significantly more complex and
expensive compared to DRAM.

Every SRAM memory cell requires the use of more than one transistor to hold its state, which takes
up a much larger physical area on the processor chip compared to the transistors used in computing
cores. This would be an important design limitation for processor designers.

Manufacturers must balance several competing factors, such as die space efficiency, production cost,
power consumption, and general performance. The larger the chip area allocated to cache memory, the
less room there is for processor cores and other functional elements. Additionally, making caches
larger increases both the production costs related to the chip and its power consumption during
operation.

Consumer central processing units, i.e., Intel Core and AMD Ryzen processors, are designed to
provide a balanced amount of performance for different computational tasks while, at the same time,
keeping the costs and power consumption within reasonable levels. Providing significant cache sizes
would make these processors very expensive and power-hungry and mostly unneeded for the average consumer.

In addition to this, most common computing operations do not benefit significantly from having very
large caches. Standard applications like web browsing, office applications, and even most games do
not require ongoing interaction with the large data structures so typical of QOA programs.

Even the highest-end and most costly workstation-class processors, such as top-of-the-line AMD
Threadripper processors, AMD EPYC server processors, and Intel Xeon workstation processors, only
have cache sizes large enough to fully contain quantum state vectors in both L2 or L3 cache. Based
on existing specifications, such processors tend to offer cache sizes of between 128 megabytes and
768 megabytes

I expect future generations of consumer CPUs could have cache sizes larger than 128 megabytes or more (10950x3d?)
in the coming years, as this would significantly increase the QOA program simulation abilities of consumer-level hardware.

## The performance impact

Quantum simulation software operates in a way that is very different from typical boolean logic
software, which is usually optimized to run on traditional CPUs. Quantum simulations involve
persistent and intensive computation throughout the whole state vector at once. When a quantum gate
operation is introduced, which occurs regularly throughout the simulation, the processor needs to
read and modify all the amplitude values.

### The following are examples of how cache size affects performance:

### up to about 16 qubits (requiring less than 8 megabytes of cache):

In the context of quantum mechanics, the state vector is quite compact and typically takes up about
1 megabyte of memory. The state vector can usually be completely held within the L3 cache of many
CPUs from 10 to 15 years ago, as long as these CPUs have large enough cache capacities. If the
entire state vector is kept in cache, the processor is able to access the quantum state data with
virtually no latency. This would allow for efficient computation and response for
end users.

The QOA program would be running at the highest computational efficiency of the processing cores.

### at about 16 qubits (which requires over 32 megabytes of cache:)

the state vector grows too large to fit within available cache memory. the cpu must now retrieve
data from the much slower system ram for every operation. this situation is called a cache miss, and
it creates a massive performance bottleneck that massively slows the speed of the computation.

The powerful cpu cores spend most of their time waiting for data to arrive from system
memory rather than performing actual calculations. the QOA program would become memory-bound and
input/output-bound rather than cpu-bound, and speed is would be limited by memory access speed
rather than the cpu's raw computational power. This explains why a lower spec CPU that has a bigger
cache can actually outperform a higher spec CPU with a smaller cache when running quantum
simulation workloads, which is not what one would expect from general benchmarking tests.

### the case for high-cache cpus (AMD Epyc/Threadripper & Intel Xeon)

If someone were to select a processor for QOA program simulation, there would be a critical need to select
appropriate processors for quantum simulation tasks. Server and high-performance desktop-specific processors,
like those in AMD's EPYC and Threadripper line, are built with different architectures than those used in
consumer-level processors. (Same ISA most of the time)

Such processors are specifically designed to deal with data-intensive computations, e.g., scientific
computing, database operations, and high-performance computing applications. Their main advantage in
the quantum simulation context is their large L3 cache, which often extends to hundreds of
megabytes.

Large cache sizes allow for direct handling of much bigger quantum state vectors to be held in the
processor chip itself. A processor with an L3 cache of 256 megabytes has the capacity to run a
simulation of about 24 qubits while having the entire state vector held in fast cache memory. This setup
avoids having to move quantum state information into slow system RAM so it stays in
processor's effective cache hierarchy.

These chips facilitate carrying out simulations of around 20 to 30 qubits and thus make them not
only possible but also operational for research and development activities. For applications where
simulation of quantum systems is beyond about 25 qubits, a central processing unit with a large
cache goes beyond simply providing a modest boost to performance; rather, it becomes a prerequisite
to avoid protracted waits of 10 minutes or longer for standard operations, an occurrence which is
due to slow performance characteristics of memory-intensive programs.

### additional characteristics and system requirements

beyond the cpu cache requirements, several other system components become critical for quantum
simulation beyond 25 qubits:

#### memory throughput and latency

When the quantum state vector exceeds cache capacity boundaries, system memory speed becomes the
main limitation on performance. Therefore, DDR5 memory implementation, which is marked by higher
speed and lower latency, is of utmost importance. In addition to this, memory bandwidth is
determined by how fast data is transferred between CPU and RAM and plays a role directly in quantum
gate operations efficiency. So for example if you have 512 gigabytes of slower MT/s DDR5 (~4800 MT/s),
it would be best to use less system ram, lets say 256 gigabytes running at 6400 MT/s, this wouldnt provide
as much preformance boost as using a higher cache CPU, but in QOA programs where 256GB of system memory
would be adequate, i'd take the faster memory speeds over raw memory capacity. 

#### The system's stability and temperature regulation

Large-scale QOA simulations can put a heavy load on computing resources. The ongoing and intensive
memory operations, together with the resulting high CPU usage, produce considerable amounts of heat.
Proper cooling systems are essential to prevent thermal throttling, which would further worsen
performance degradation. 

This seems pretty self-explanantory but im sure there would be a idiot out there trying to run a 30 qubit
QOA program on a intel stock cooler, so I included this anyways.

#### operating systems and memory resource administration

The ability of an operating system to manage memory is critical for running large simulations. Some
operating systems perform better in handling large memory allocations than others Additionally,
virtual memory systems can impact performance negatively, while memory fragmentation problems can
prevent optimal allocation of the large contiguous blocks of memory needed for quantum state
vectors.

Some reccomendations for Linux distrubutions of mine include Ubuntu (or Ubuntu server), RHEL, and Debian 12/13.
These seem to be best for large memory management and have been tried & tested over the years, including by me.

QOA is not limited to just Linux, it can run whereever Cargo (rust compliler), can compile.
so that can include Windows, BSD, MacOS, even Android probably (if done correctly.)

## Summary & My Recommendations

The fundamental design of modern computing systems presents serious practical limitations to
QOA simulation via classical methods. In particular, the memory hierarchy, and notably the sizes of CPU
caches, constitutes the main limitation instead of raw computational power.

In the area of consumer-level hardware with respect to commodities, a reasonable ceiling for quantum
simulation is about 25 qubits. Attempts to exceed this threshold will cause memory requirements to
increase exponentially, first consuming the CPU cache, then the system's RAM, and eventually causing
instability in the system, and in worst-case scenarios, system crashing and loss of everything in volatile memory (RAM).

These challenges to scalability cannot be overcome by advances either in software or algorithms but
are rooted in intrinsic properties of the mathematical framework on which quantum systems are based
and in fundamental limitations of classical computer hardware.

Sure, I can optimize memory usage, but *n* bytes of memory required will still double for each qubit added.

For large-scale QOA simulations activities that require the use of more than 25 qubits, it is
reccomended to use specialized hardware setups. This need involves the procurement of systems with
large amounts of fast memory and, more importantly, processors with very large cache sizes.

Without the appropriate hardware, the QOA programs you run can suffer from significant delays, cause system
instability, and possibly never complete and crash your system. It is recommended for those involved in running QOA
programs to analyze thoroughly their system's cache and memory architecture to determine the possible number
of qubits that can be supported before hitting these limitations. 

### Thanks for reading
#### --Rayan
