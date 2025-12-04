import numpy as np
import csv
import matplotlib.pyplot as plt

total_data_avg = []
with open("size_data.csv", newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        total_data_avg.append(float(row[0]))

avg_size = np.mean(total_data_avg)
std_size = np.std(total_data_avg)

x = np.arange(1, len(total_data_avg)+1)
y = np.array(total_data_avg)

plt.title("Size per Detection")
plt.xlabel("Detection(frame)")
plt.ylabel("Size(cm)")
#plt.scatter(x, y, color="green")
plt.plot(x, y, color="green")
plt.text(int(len(total_data_avg)*0.7), 0.8, f"avg: {avg_size:.2f}, std: {std_size:.2f}")
plt.axis((-0.05*len(total_data_avg), 1.05*len(total_data_avg), 0, 1.0))

plt.savefig("size_data_dynamic_10.jpg")
plt.show()

