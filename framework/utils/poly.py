''' pixmatch gta5 config
opt:
  kind: "SGD"
  momentum: 0.9
  weight_decay: 5e-4
  lr: 1e-4
  iterations: 40000
  poly_power: 0.9
'''

def poly_lr_scheduler(args, iter, optimizer):
    init_lr = args.init_lr
    iter = iter
    max_iter = args.lr_max_iter
    power = args.poly_power
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]["lr"] = 10 * new_lr