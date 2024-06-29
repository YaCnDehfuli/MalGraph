# Malware Detection using Memory Dump

Research project: detect and classify malware from a **runtime memory snapshot** of a
process, instead of the on-disk binary. Capturing memory can expose unpacked / decrypted /
injected code that the disk executable hides.

Rough pipeline (work in progress):

1. Run the sample in an isolated VM, dump the VM memory (`Run.sh`).
2. Carve the process out of the dump with Volatility, disassemble it with SMDA.
3. Build per-function control-flow graphs from the disassembly.
4. Learn instruction/basic-block representations, then classify.

More to come.
