import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import pandas as pd

user_name = r"C:\Users\emago\OneDrive\Escritorio\Code_and_Design"
# Load the image of the PCB board 
source_file = user_name + r"/Evaluation/rect_8.png"
## choose the results of the board that want to be check in the board 

csv_file_path = user_name + r"/.Results\rect_2decap_del\Qmix_termination_state_152_R.csv"

img = mpimg.imread(source_file)
# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
#######################################################################################
##to get the coordinates of a new images use the code bellow, Use this part to select the position 
## of each decap on the board from the figure get it in the source_file 
#######################################################################################
# coords = []

# def onclick(event):
#     ix, iy = event.xdata, event.ydata
#     coords.append((ix, iy))
#     print(f'Coordinates: ({ix}, {iy})')
#     # Add a red dot where you clicked
#     plt.plot(ix, iy, 'ro')
#     plt.draw()

# # Display the image and set up the click event
# fig, ax = plt.subplots()
# ax.imshow(img)
# cid = fig.canvas.mpl_connect('button_press_event', onclick)

# plt.show()

# # After clicking, print all collected coordinates
# print('All coordinates:', coords)
#########################################################################################
# Coordinates for each port 
#Save the coordinates from the previous step in this c
###########################################################################################

# Coordinates for 12 square
# port_coords = {
#     'C1': (135.0194805194805, 120.90259740259734),
#     'C2': (267.03246753246754, 119.6688311688311),
#     'C3': (403.98051948051943, 119.6688311688311),
#     'C4': (132.55194805194805, 198.62987012987008),
#     'C5': (269.5, 199.86363636363632),
#     'C6': (405.21428571428567, 201.09740259740255),
#     'C7': (133.78571428571428, 281.2922077922077),
#     'C8': (268.26623376623377, 278.82467532467524),
#     'C9': (402.7467532467532, 280.0584415584415),
#     'C10': (133.78571428571428, 349.1493506493506),
#     'C11': (269.5, 349.1493506493506),
#     'C12': (406.4480519480519, 350.3831168831168)
# }

# Coordinates for 12 rect 
# port_coords = {
#     'C1': (87.68548387096774, 99.56451612903214),
#     'C2': (179.1370967741935, 100.65322580645153),
#     'C3': (270.5887096774194, 99.56451612903214),
#     'C4': (358.7741935483871, 100.65322580645153),
#     'C5': (88.77419354838707, 145.29032258064507),
#     'C6': (179.1370967741935, 149.6451612903225),
#     'C7': (267.3225806451612, 149.6451612903225),
#     'C8': (359.86290322580646, 150.7338709677418),
#     'C9': (88.77419354838707, 199.7258064516128),
#     'C10': (176.95967741935485, 200.81451612903217),
#     'C11': (269.5, 201.90322580645153),
#     'C12': (357.6854838709677, 201.90322580645153)
# }

# ##Coodinates 8 sqr
# port_coords = {
#     'C1': (174.5, 118.43506493506487),
#     'C2': (334.88961038961037, 119.6688311688311),
#     'C3': (175.73376623376623, 198.62987012987008),
#     'C4': (334.88961038961037, 201.09740259740255),
#     'C5': (173.26623376623377, 277.590909090909),
#     'C6': (337.35714285714283, 282.52597402597394),
#     'C7': (174.5, 352.8506493506493),
#     'C8': (336.1233766233766, 350.3831168831168)
# }

##Coodinates 8 sqr
port_coords = {
    'C1': (104.01612903225805, 131.13709677419348),
    'C2': (192.20161290322577, 134.40322580645153),
    'C3': (285.83064516129025, 128.95967741935476),
    'C4': (372.92741935483866, 131.1370967741934),
    'C5': (104.01612903225805, 221.49999999999991),
    'C6': (193.29032258064515, 220.41129032258058),
    'C7': (285.83064516129025, 219.32258064516122),
    'C8': (374.01612903225805, 219.32258064516122)
}


def plot_board(active_ports):
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Plot each port
    for i, (port, coord) in enumerate(port_coords.items()):
        if active_ports[i] == 1:
            # Draw a green circle over the active port
            circle = plt.Circle((coord[0], coord[1]), 10, color='green',label='Type 1', fill=True)
            ax.add_patch(circle)
        if active_ports[i] == 2:
            # Draw a green circle over the active port
            circle = plt.Circle((coord[0], coord[1]), 10, color='red', label= 'Type 2',fill=True)
            ax.add_patch(circle)

    handles, labels = ax.get_legend_handles_labels()
    unique_handles = {label: handle for label, handle in zip(labels, handles)}.values()
    ax.legend(handles=unique_handles, loc='upper right', fontsize='x-large')

    plt.axis('off')
    plt.show()

### get the optimal position from the CSV file #####
states = df['state']
rewards = df['rewards']
# Convert states from string to list
states = states.apply(lambda x: tuple(eval(x.replace(' ', ', '))))
rewards = rewards.apply(lambda x: max(map(float, eval(x).values())))

# Count the frequency of each state
state_counter = Counter(states)

# Find the most common state
most_common_state = state_counter.most_common(1)[0]
best_reward_index = rewards.idxmax()
best_reward_state = states[best_reward_index]
best_reward_value = rewards[best_reward_index]

#### print the values of the positions and show the physical position 
print(f"The most frequent state is {most_common_state[0]} which appears {most_common_state[1]} times.")
print(f"The state with the best reward is {best_reward_state} with a reward of {best_reward_value}.")

plot_board(most_common_state[0])