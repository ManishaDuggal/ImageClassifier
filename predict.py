import myfunctions
import json
import torch
import sys

# Fetch the command line arguments
args = myfunctions.get_inp_args_predict()

# Load the class to name .json file
category_names = args.category_names
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the saved model
checkpoint_path = args.checkpoint
model = myfunctions.load_model(checkpoint_path)

gpu = args.gpu
device = torch.device("cuda:0" if (gpu and torch.cuda.is_available()) else "cpu")

if (gpu and not torch.cuda.is_available()):
    sys.exit("GPU not available!")

# Predict classes & probabilities
top_k = args.top_k
img_path = args.img_path
top_k_values, classes = myfunctions.predict(img_path, model, top_k, device)

# Get names from classes
names = [cat_to_name[str(index)] for index in classes]

print ("The flower is {} with probability of {:.2f}.".format(names[0], top_k_values[0]))
print("\nOther predictions -")
for i in range(1, top_k):
    print("{}         {:.2f}".format(names[i], top_k_values[i]))
