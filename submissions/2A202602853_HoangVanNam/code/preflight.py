"""Overfit a fixed batch, checking both training and evaluation modes."""
import torch


def overfit_batch(model, x, y, max_steps=100, lr=.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    history = []
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x)
            eval_loss = float(criterion(logits, y))
            accuracy = float((logits.argmax(1) == y).float().mean())
        history.append({'step': step, 'train_loss': float(loss.detach()), 'eval_loss': eval_loss})
        if float(loss.detach()) < .02 and eval_loss < .05 and accuracy == 1.:
            break
    return {'steps': step, 'train_loss': float(loss.detach()), 'eval_loss': eval_loss,
            'accuracy': accuracy, 'history': history}
