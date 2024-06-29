# Lab setup

The capture side of this project runs against a throwaway Windows 10 guest under
KVM/QEMU (libvirt). Notes to my future self so I can rebuild it.

## Isolation
- The guest sits on an **isolated** libvirt network (no route to the host LAN or the
  internet). Only the host can reach it over SSH for launching the sample.
- Every run ends with a revert to the `clean_state` snapshot, so the guest never
  accumulates state between samples.

## Snapshot
```
virsh snapshot-create-as --domain VM1_win10 clean_state "clean baseline"
```
`Run.sh` reverts to this after each capture.

## Capture
- The sample is started with `Start-Process` over SSH, given a short dwell time, then the
  **whole-VM** memory is dumped with `virsh dump --memory-only`.
- The dwell time is intentionally short for now; long-running samples need a longer window.

## Credentials
- The guest password is read from `$VM_PASSWORD` in the environment. Nothing secret is
  committed.

## Prereqs on the host
- libvirt + qemu-kvm, `virsh`, `sshpass`
- Volatility 3 (for carving the process out of the dump)
- SMDA (for disassembly)
