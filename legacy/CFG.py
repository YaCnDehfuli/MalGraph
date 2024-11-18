from smda.Disassembler import Disassembler
import json
from smda.SmdaConfig import SmdaConfig
import networkx as nx
import matplotlib.pyplot as plt

# Create SMDA configuration and set architecture
config = SmdaConfig()
config.architecture = "intel.64bit"  # Set to "intel.64bit" if your process is 64-bit

dis_obj=Disassembler(config)
print("generating report")
smda_report = dis_obj.disassembleFile("/home/yacn/combined_6280.dmp")
# #0x990878c06d60.0x990877edf050.ImageSectionObject.svchost.exe-2.img

# # report = dis_obj.disassembleFile("/home/yacn/combined_memory_dump.img")
# # report = dis_obj.disassembleFile("/home/yacn/file.0x990878c06d60.0x990877edf050.ImageSectionObject.svchost.exe-2.img")

# print("report generated")
# print(smda_report)
# json_report = smda_report.toDict()    
# output_file = "/home/yacn/Data/combined_6280.json"
# with open(output_file, 'w') as f:
#     json.dump(json_report, f, indent=4)

# Sample function to construct CFG from SMDA report
def construct_cfg(smda_report):
    G = nx.DiGraph()  # Use a directed graph for control flow

    # Assuming 'functions' is a key in the smda_report dictionary
    for function in smda_report['functions']:
        func_name = function['name']
        blocks = function['basic_blocks']

        # Adding nodes (basic blocks) to the graph
        for block in blocks:
            block_address = block['address']
            G.add_node(block_address, label=f"Block @ {hex(block_address)}")

            # Adding edges based on control flow
            for succ in block['successors']:
                G.add_edge(block_address, succ)

    return G

# Sample function to visualize CFG using matplotlib
def visualize_cfg(G):
    pos = nx.spring_layout(G)  # Position nodes for a clearer layout
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.title("Control Flow Graph")
    plt.show()

# Load your smda report JSON here

with open("/home/yacn/Data/combined_6280.json") as file:
    test_report = json.load(file)

# Construct and visualize CFG
cfg_graph = construct_cfg(test_report)
visualize_cfg(cfg_graph)



# python3 ~/volatility3/vol.py -f ~/Research/dumps/wmplayer.raw -o /home/yacn/Data/ windows.memmap.Memmap --pid 1340 --dump
