import torch
import time
from utils.custom_dataset import TokenizedTextDataset
from utils.metrics import MetricsWriter, Metrics
from tqdm import tqdm


def train_step(model, loss_fn, opt, loader):
    loss_per_batches = 0
    start_epoch2 = time.time()

    for i, data in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        input_ids = data["input_ids"].to(model.device)
        attention_mask = data["attention_mask"].to(model.device)
        token_type_ids = (data["token_type_ids"].to(model.device)
                          if "token_type_ids" in data else None) 
        labels = data["labels"].to(model.device)
        opt.zero_grad()

        y_pred = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_per_batches += loss.item()

    print("train + load = " + str(time.time() - start_epoch2))
    return loss_per_batches/(i+1)  

def train(model, mw, loss_fn, opt, train_loader, val_loader, epochs=20):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        mw.print_epoch(epoch + 1)
        metrics_valid = Metrics()

        model.train()
        avg_loss = train_step(model, loss_fn, opt, train_loader)
        model.eval()

        vloss = 0
        counter = 0
        with torch.inference_mode():
            for i, vdata in enumerate(val_loader):
                vinput_ids = vdata["input_ids"].to(model.device)
                vattention_mask = vdata["attention_mask"].to(model.device)
                vtoken_type_ids = (vdata["token_type_ids"].to(model.device)
                                  if "token_type_ids" in vdata else None)
                vlabels = vdata["labels"].to(model.device)
                
                y_pred = model(vinput_ids, vtoken_type_ids, vattention_mask)
                bloss = loss_fn(y_pred, vlabels)
                vloss += bloss.item()

                probs = torch.sigmoid(y_pred)
                y_pred = (probs > 0.5).long()
                metrics_valid.batch_step(vlabels, y_pred)
                counter = i

        avg_vloss = vloss / (counter + 1)

        scheduler.step()

        valrecall, valprecision, valf1, valacc = metrics_valid.get_metrics()
        mw.writer_step(avg_loss, avg_vloss, valrecall, valprecision, valf1, valacc)
        mw.save_model(model)
        mw.print_time()