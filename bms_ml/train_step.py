def train_one_batch(model, x, y, loss_fn, optimizer):
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()