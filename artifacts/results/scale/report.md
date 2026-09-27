# What breaks at 10× and 100× — measured

`scripts/measure_scale.py` at `2a1e787`; each case is one single-threaded process on a machine with 12 usable CPUs.

| case | policy | requests | block hashes | generate | simulate | peak RSS |
|---|---|---:|---:|---:|---:|---:|
| 1× trace | round-robin | 800 | 84,600 | 0.8 s | 0.1 s | 29 MiB |
| 1× trace | radix-affinity-balanced | 800 | 84,600 | 0.8 s | 0.2 s | 29 MiB |
| 10× trace | round-robin | 8,000 | 846,000 | 8.6 s | 1.3 s | 76 MiB |
| 10× trace | radix-affinity-balanced | 8,000 | 846,000 | 8.1 s | 2.0 s | 81 MiB |
| 100× trace | round-robin | 80,000 | 8,460,000 | 86.8 s | 13.5 s | 549 MiB |
| 100× trace | radix-affinity-balanced | 80,000 | 8,460,000 | 85.5 s | 21.0 s | 571 MiB |
| 10× replicas (and 10× load) | round-robin | 8,000 | 846,000 | 7.7 s | 1.4 s | 128 MiB |
| 10× replicas (and 10× load) | radix-affinity-balanced | 8,000 | 846,000 | 8.5 s | 4.8 s | 129 MiB |
