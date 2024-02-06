from transformers.integrations import WandbCallback

def decode_predictions(predictions):
    labels = predictions.label_ids  # Use integer labels directly
    logits = predictions.predictions.argmax(axis=-1)
    return {"labels": labels, "predictions": logits}


class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset, df_text):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.df_text = df_text
    
    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control, **kwargs)

        # Generate predictions and process them
        raw_predictions = self.trainer.predict(self.val_dataset)
        decoded_predictions = decode_predictions(raw_predictions)

        # Creating DataFrame and mapping labels
        label_dict = {0: 'IT과학', 1: '경제', 2: '사회', 3: '생활문화', 4: '세계', 5: '스포츠', 6: '정치'}
        predictions_df = pd.DataFrame(decoded_predictions)
        predictions_df['texts'] = self.df_text['text']
        # predictions_df['datetime'] = self.df_text['datetime']
        predictions_df['isCorrect'] = predictions_df.apply(lambda row: 'Yes' if row['labels'] == row['predictions'] else 'No', axis=1)
        predictions_df['labels'] = predictions_df['labels'].map(label_dict)
        predictions_df['predictions'] = predictions_df['predictions'].map(label_dict)
        

        # Logging Precision-Recall Curve and Confusion Matrix to wandb
        self._wandb.log({
            "precision_recall_curve": wandb.plot.pr_curve(y_true=predictions_df['labels'], y_probas=raw_predictions.predictions, labels=list(label_dict.values()))
        })

        self._wandb.log({
            "confusion_matrix": wandb.sklearn.plot_confusion_matrix(predictions_df['labels'], predictions_df['predictions'], list(label_dict.values()))
        })

        # Create and log wandb.Table
        records_table = self._wandb.Table(dataframe=predictions_df)
        self._wandb.log({"predictions": records_table})
