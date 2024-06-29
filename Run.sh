#!/bin/bash
# Launch a sample inside the isolated analysis VM and capture its memory.
# Credentials come from the environment, never hard-coded:
#   export VM_PASSWORD=... before running.

VM_Name="VM1_win10"
VM_IP="192.168.124.51"
VM_Username="bcccu"
VM_Password="${VM_PASSWORD:?set VM_PASSWORD in the environment}"
Malware_Path="C:\\Program Files\\Windows Media Player\\wmplayer.exe"
Dump_Path="/home/yacn/Research/dumps/"
Malware_Name="wmplayer"

Start_SSH="sshpass -p '$VM_Password' ssh -o StrictHostKeyChecking=no ${VM_Username}@${VM_IP}"
Run_exe="powershell -Command Start-Process '$Malware_Path'"

# start the sample in the guest
$Start_SSH "$Run_exe"

echo "Running $Malware_Name for a short window before capture"
sleep 10

echo "Dumping the memory of $VM_Name"
virsh -c qemu:///system dump --memory-only --file "$Dump_Path$Malware_Name.raw" "$VM_Name"

echo "Reverting $VM_Name to its clean snapshot"
virsh -c qemu:///system snapshot-revert --domain "$VM_Name" --snapshotname clean_state --force

echo "Done, dumped to $Dump_Path"

