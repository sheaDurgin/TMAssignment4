# TMAssignment4

This code contains files that fine-tune a distilroberta-base model on genre to lyrics data
There are scripts that test, epochs, batch_size, and weight_decay values separately
Then a final script that uses the best values from each to fine-tune a final model
Each script reports f1-scores on the test set after running

step2-final.py is the only script with an argument,
`python step2-final.py train` will fine-tune a model on the chosen hyperparameters and predict on test set
`python step2-final.py` will use the default pre-trained model for prediction