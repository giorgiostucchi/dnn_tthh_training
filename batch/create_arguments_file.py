LR = [0.005, 0.01]
BATCHSIZE = [128, 256]
NODE = [12, 24]
LAYER = [3, 4]
DROPOUT = [0.1, 0.3]

parameter_combinations = []

for lr in LR:
    for bs in BATCHSIZE:
        for node in NODE:
            for layer in LAYER:
                for dropout in DROPOUT:
                    parameter_combinations.append((lr, bs, node, layer, dropout))

# Create and write to arguments.txt
with open('arguments.txt', 'w') as file:
    for params in parameter_combinations:
        argument_line = ' '.join(map(str, params))
        file.write(argument_line + '\n')